import contextlib
import os
import signal
import time

import psutil
import pytest
import ray
from ray.actor import ActorHandle
from ray.util.state import list_actors

from polars_nanook_ray.identity import SCHEDULER

from .conftest import RayClusterFactory, TestQuery, worker_actors


def _actor_pid(cluster_id: str, name: str) -> int:
    for actor in list_actors(
        filters=[
            ("ray_namespace", "=", cluster_id),
            ("state", "=", "ALIVE"),
        ]
    ):
        if actor.name == name and actor.pid is not None:
            return actor.pid

    msg = f"Actor '{name!r}' not found in namespace '{cluster_id!r}'"
    raise AssertionError(msg)


def _wait_for_worker_restart(
    cluster_id: str,
    name: str,
    actor: ActorHandle,
    actor_pid: int,
) -> int:
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        try:
            new_actor_pid = _actor_pid(cluster_id, name)
        except AssertionError:
            new_actor_pid = None

        if new_actor_pid is not None and new_actor_pid != actor_pid:
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                if ray.get(actor.is_ready.remote()):
                    return new_actor_pid
                time.sleep(1)

        time.sleep(1)

    msg = "New actor did not restart properly"
    raise RuntimeError(msg)


def test_worker_restart_cleanup(
    ray_cluster: RayClusterFactory,
    run_query: TestQuery,
) -> None:
    cluster = ray_cluster(num_workers=1)
    cluster_id = cluster.config.cluster_id

    ((worker, actor),) = worker_actors(cluster).items()
    name = worker.actor_name
    actor_pid = _actor_pid(cluster_id, name)
    binary_pid = ray.get(actor.get_worker_pid.remote())

    # killing the actor will orphan the child process
    os.kill(actor_pid, signal.SIGKILL)
    time.sleep(2)

    try:
        new_actor_pid = _wait_for_worker_restart(cluster_id, name, actor, actor_pid)
        new_binary_pid = ray.get(actor.get_worker_pid.remote())

        assert new_actor_pid is not None
        assert new_actor_pid != actor_pid

        assert new_binary_pid != binary_pid
        assert not psutil.pid_exists(binary_pid)

        # the restarted worker takes over from the one it replaces
        run_query(cluster=cluster)
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.kill(binary_pid, signal.SIGKILL)


def test_worker_binary_death_restarts_its_actor(
    ray_cluster: RayClusterFactory,
    run_query: TestQuery,
) -> None:
    cluster = ray_cluster(num_workers=1)
    cluster_id = cluster.config.cluster_id

    ((worker, actor),) = worker_actors(cluster).items()
    name = worker.actor_name
    actor_pid = _actor_pid(cluster_id, name)
    binary_pid = ray.get(actor.get_worker_pid.remote())

    # the actor stays up, only its binary dies, e.g. by the OOM killer
    os.kill(binary_pid, signal.SIGKILL)

    new_actor_pid = _wait_for_worker_restart(cluster_id, name, actor, actor_pid)

    assert new_actor_pid != actor_pid
    assert ray.get(actor.get_worker_pid.remote()) != binary_pid
    run_query(cluster=cluster)


@pytest.mark.parametrize("kill", ["actor", "binary"])
def test_scheduler_restarts_and_workers_register_again(
    ray_cluster: RayClusterFactory,
    run_query: TestQuery,
    kill: str,
) -> None:
    cluster = ray_cluster(num_workers=1)
    cluster_id = cluster.config.cluster_id
    scheduler_actor = cluster._scheduler_actor

    sched_pid = _actor_pid(cluster_id, SCHEDULER.actor_name)
    binary_pid = ray.get(scheduler_actor.get_scheduler_pid.remote())

    # the actor's death orphans the binary, the binary's death exits the actor
    os.kill(sched_pid if kill == "actor" else binary_pid, signal.SIGKILL)
    if kill == "actor":
        with contextlib.suppress(ProcessLookupError):
            os.kill(binary_pid, signal.SIGKILL)

    # Ray restarts the scheduler on the head, with no cluster.start() needed
    deadline = time.monotonic() + 60
    new_sched_pid = None
    while time.monotonic() < deadline:
        with contextlib.suppress(AssertionError):
            new_sched_pid = _actor_pid(cluster_id, SCHEDULER.actor_name)
            if new_sched_pid != sched_pid:
                break
        time.sleep(1)
    assert new_sched_pid not in (None, sched_pid)

    # allow some retries for the worker to register again
    result = None
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        try:
            result = run_query(cluster=cluster)
            break
        except Exception:
            time.sleep(2)

    assert result is not None
