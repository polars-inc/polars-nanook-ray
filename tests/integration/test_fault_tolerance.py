import contextlib
import os
import signal
import socket
import time

import psutil
import pytest
import ray
from ray.actor import ActorHandle
from ray.util.state import list_actors

from polars_nanook_ray import Ports
from polars_nanook_ray.identity import SCHEDULER

from .conftest import (
    ContextFactory,
    TestQuery,
    name_of,
    scheduler_actor,
    worker_actors,
)


def _actor_pid(namespace: str, name: str) -> int:
    for actor in list_actors(
        filters=[
            ("ray_namespace", "=", namespace),
            ("state", "=", "ALIVE"),
        ]
    ):
        if actor.name == name and actor.pid is not None:
            return actor.pid

    msg = f"Actor '{name!r}' not found in namespace '{namespace!r}'"
    raise AssertionError(msg)


def _wait_for_worker_restart(
    namespace: str,
    name: str,
    actor: ActorHandle,
    actor_pid: int,
) -> int:
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        try:
            new_actor_pid = _actor_pid(namespace, name)
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
    ray_cluster: ContextFactory,
    run_query: TestQuery,
) -> None:
    cluster = ray_cluster(workers=1)
    namespace = name_of(cluster)

    ((worker, actor),) = worker_actors(cluster).items()
    name = worker.actor_name
    actor_pid = _actor_pid(namespace, name)
    binary_pid = ray.get(actor.get_worker_pid.remote())

    # the binary dies with its actor's process
    os.kill(actor_pid, signal.SIGKILL)
    time.sleep(2)

    try:
        new_actor_pid = _wait_for_worker_restart(namespace, name, actor, actor_pid)
        new_binary_pid = ray.get(actor.get_worker_pid.remote())

        assert new_actor_pid is not None
        assert new_actor_pid != actor_pid

        assert new_binary_pid != binary_pid
        assert not psutil.pid_exists(binary_pid)

        # the restarted worker takes over from the one it replaces
        run_query(cluster)
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.kill(binary_pid, signal.SIGKILL)


def test_worker_binary_death_restarts_its_actor(
    ray_cluster: ContextFactory,
    run_query: TestQuery,
) -> None:
    cluster = ray_cluster(workers=1)
    namespace = name_of(cluster)

    ((worker, actor),) = worker_actors(cluster).items()
    name = worker.actor_name
    actor_pid = _actor_pid(namespace, name)
    binary_pid = ray.get(actor.get_worker_pid.remote())

    # the actor stays up, only its binary dies, e.g. by the OOM killer
    os.kill(binary_pid, signal.SIGKILL)

    new_actor_pid = _wait_for_worker_restart(namespace, name, actor, actor_pid)

    assert new_actor_pid != actor_pid
    assert ray.get(actor.get_worker_pid.remote()) != binary_pid
    run_query(cluster)


@pytest.mark.parametrize("kill", ["actor", "binary"])
def test_scheduler_restarts_and_workers_register_again(
    ray_cluster: ContextFactory,
    run_query: TestQuery,
    kill: str,
) -> None:
    # a fixed client port, for the context to reach the restarted scheduler
    with socket.socket() as sock:
        sock.bind(("", 0))
        client_port = sock.getsockname()[1]
    cluster = ray_cluster(workers=1, ports=Ports(client=client_port, dashboard=None))
    namespace = name_of(cluster)
    scheduler = scheduler_actor(cluster)

    sched_pid = _actor_pid(namespace, SCHEDULER.actor_name)
    binary_pid = ray.get(scheduler.get_scheduler_pid.remote())

    # the binary dies with the actor, and the actor exits with the binary
    os.kill(sched_pid if kill == "actor" else binary_pid, signal.SIGKILL)

    # Ray restarts the scheduler on the head, with no start() needed
    deadline = time.monotonic() + 60
    new_sched_pid = None
    while time.monotonic() < deadline:
        with contextlib.suppress(AssertionError):
            new_sched_pid = _actor_pid(namespace, SCHEDULER.actor_name)
            if new_sched_pid != sched_pid:
                break
        time.sleep(1)
    assert new_sched_pid not in (None, sched_pid)

    # allow some retries for the worker to register again
    result = None
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        try:
            result = run_query(cluster)
            break
        except Exception:
            time.sleep(2)

    assert result is not None
