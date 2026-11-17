"""The scheduler actor's scaling endpoint, engine and refill thread stop together.

They stop however the actor stops, and come back with it when Ray restarts it.

Uses its own Ray cluster and stub engines, so no license is needed.
"""

import json
import os
import pathlib
import signal
import socket
import time
import typing
import urllib.request

import psutil
import pytest
import ray
from ray.actor import ActorHandle
from ray.cluster_utils import Cluster
from ray.exceptions import ActorUnavailableError, RayActorError

from polars_nanook_ray.config import (
    PolarsObservatoryConfig,
    PolarsRayClusterConfig,
    PolarsScalingConfig,
    PolarsSchedulerConfig,
)
from polars_nanook_ray.identity import list_workers
from polars_nanook_ray.ports import resolve_ports

from .utils import (
    free_port,
    head_cluster,
    make_config,
    run_scheduler,
    wait_for_workers,
)


@pytest.fixture(scope="module")
def cluster() -> typing.Iterator[Cluster]:
    cluster = head_cluster()
    cluster.add_node(num_cpus=2)
    cluster.wait_for_nodes()

    yield cluster

    ray.shutdown()
    cluster.shutdown()


@pytest.fixture
def port() -> int:
    return resolve_ports({"scaling.port": None})["scaling.port"]


def scaling_config(port: int, **kwargs: typing.Any) -> PolarsRayClusterConfig:
    scheduler = PolarsSchedulerConfig(
        cpus_hint=0,
        memory_hint=1,
        client_port=free_port(),
        worker_registration_port=free_port(),
        observatory=PolarsObservatoryConfig(enabled=False),
        scaling=PolarsScalingConfig(enabled=True, port=port),
    )
    return make_config(scheduler=scheduler, **kwargs)


def scale_config(port: int) -> dict[str, typing.Any]:
    url = f"http://127.0.0.1:{port}/scale_config"
    with urllib.request.urlopen(url, timeout=5) as response:
        return json.loads(response.read())


def wait_until(condition: typing.Callable[[], bool], timeout: float = 30) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.2)


def is_free(port: int) -> bool:
    """Return whether a new scheduler could listen on `port`."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", port))
            sock.listen()
        except OSError:
            return False
        return True


def is_gone(pid: int) -> bool:
    try:
        return psutil.Process(pid).status() == psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return True


def actor_pid(actor: ActorHandle) -> int:
    pid = ray.get(  # type: ignore[var-annotated]
        actor.__ray_call__.remote(lambda _: os.getpid()), timeout=30
    )
    return typing.cast("int", pid)


def restarted(actor: ActorHandle, pid: int) -> bool:
    try:
        return actor_pid(actor) != pid
    except ActorUnavailableError:  # restarting
        return False


def engine_pid(actor: ActorHandle) -> int:
    return typing.cast("int", ray.get(actor.get_scheduler_pid.remote(), timeout=30))


Scheduler = typing.Callable[..., ActorHandle]


@pytest.fixture
def scheduler(cluster: Cluster) -> typing.Iterator[Scheduler]:  # noqa: ARG001
    """Start a scheduler actor from a config, and stop it after the test."""
    running: list[typing.Iterator[ActorHandle]] = []

    def _start(config: PolarsRayClusterConfig, max_restarts: int = 0) -> ActorHandle:
        running.append(run_scheduler(config, max_restarts))
        return next(running[-1])

    yield _start

    for actor in running:
        next(actor, None)


def started(scheduler: Scheduler, config: PolarsRayClusterConfig) -> ActorHandle:
    """Return a scheduler once its engine runs and its scaling endpoint answers."""
    actor = scheduler(config)
    engine_pid(actor)
    assert scale_config(config.scheduler.scaling.port)["min"] == 0
    return actor


def test_the_engine_starts_once_the_endpoint_answers(
    scheduler: Scheduler, port: int, tmp_path: pathlib.Path
) -> None:
    answer = tmp_path / "scale_config.json"
    engine = tmp_path / "engine"
    # exits, taking its actor down, unless the endpoint answers at its startup
    engine.write_text(
        "#!/bin/sh\n"
        """uri=$(sed -n 's/^uri = "\\(.*\\)"$/\\1/p' "$3")\n"""
        f'curl -sf "$uri/scale_config" > "{answer}" || exit 1\n'
        "exec sleep 3600\n"
    )
    engine.chmod(0o755)

    actor = started(scheduler, scaling_config(port, binary_path=str(engine)))

    wait_until(lambda: answer.exists() and answer.stat().st_size > 0)
    assert json.loads(answer.read_text())["desired"] == 0
    assert not is_gone(engine_pid(actor))


def test_stop_closes_the_endpoint_and_stops_the_workers_and_engine(
    scheduler: Scheduler, port: int
) -> None:
    config = scaling_config(port, num_workers=1)
    actor = started(scheduler, config)
    engine = engine_pid(actor)
    wait_for_workers(config.cluster_id, 1)

    ray.get(actor.stop.remote(), timeout=30)

    assert is_free(port)
    assert is_gone(engine)
    assert list_workers(config.cluster_id) == set()


@pytest.mark.parametrize("how", ["sigterm", "ray"])
def test_a_stopped_actor_frees_the_port_and_stops_the_engine(
    scheduler: Scheduler, port: int, how: str
) -> None:
    actor = started(scheduler, scaling_config(port))
    engine = engine_pid(actor)

    if how == "sigterm":
        os.kill(actor_pid(actor), signal.SIGTERM)
    else:  # a graceful exit, which runs __ray_shutdown__
        actor.__ray_terminate__.remote()

    wait_until(lambda: is_free(port) and is_gone(engine))


def test_a_restarted_actor_serves_on_the_same_port(
    scheduler: Scheduler, port: int
) -> None:
    actor = scheduler(scaling_config(port), max_restarts=1)
    engine = engine_pid(actor)
    pid = actor_pid(actor)
    scale_config(port)  # leaves a connection in TIME_WAIT on the port

    os.kill(pid, signal.SIGKILL)

    wait_until(lambda: is_gone(engine))
    wait_until(lambda: restarted(actor, pid))
    assert scale_config(port)["desired"] == 0


def test_a_taken_scaling_port_fails_the_start(scheduler: Scheduler, port: int) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", port))
        listener.listen()

        actor = scheduler(scaling_config(port))

        with pytest.raises(RayActorError, match=rf"scaling.port {port} .*in use"):
            ray.get(actor.get_host.remote(), timeout=30)


def test_a_failed_engine_start_frees_the_port(scheduler: Scheduler, port: int) -> None:
    actor = scheduler(scaling_config(port, binary_path="/nonexistent/engine"))

    with pytest.raises(RayActorError, match="nonexistent"):
        ray.get(actor.get_host.remote(), timeout=30)
    assert is_free(port)


def test_a_taken_engine_port_leaves_the_scaling_port_free(
    scheduler: Scheduler, port: int
) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("", 0))
        listener.listen()
        taken = listener.getsockname()[1]
        config = scaling_config(port)
        config = config.model_copy(
            update={
                "scheduler": config.scheduler.model_copy(update={"client_port": taken})
            }
        )

        actor = scheduler(config)

        with pytest.raises(RayActorError, match=rf"client_port {taken} .*in use"):
            ray.get(actor.get_host.remote(), timeout=30)
    assert is_free(port)
