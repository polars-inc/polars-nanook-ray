"""Worker placement on a simulated multi-node Ray cluster.

Several raylets run on this one machine, without the dashboard. The binaries are
replaced by a stub that sleeps, so no license is needed: only the actors are under test.
"""

import contextlib
import socket
import time
import typing

import pytest
import ray
from ray._private.state import available_resources_per_node
from ray.actor import ActorHandle
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from polars_nanook_ray.actors import WorkerExited, WorkerSlot
from polars_nanook_ray.actors.worker import RayWorkers
from polars_nanook_ray.cluster import PolarsRayCluster
from polars_nanook_ray.config import (
    PolarsObservatoryConfig,
    PolarsRayClusterConfig,
    PolarsScalingConfig,
    PolarsSchedulerConfig,
    PolarsWorkerConfig,
)
from polars_nanook_ray.identity import SCHEDULER, list_workers

from .utils import (
    FAILING_ENGINE,
    LISTENING_ENGINE,
    free_port,
    head_cluster,
    make_config,
    node_id,
    run_scheduler,
    wait_for_workers,
)

NUM_CPU_NODES = 2


@pytest.fixture(scope="module")
def cpu_node_ids() -> typing.Iterator[set[str]]:
    cluster = head_cluster()
    nodes = [cluster.add_node(num_cpus=2) for _ in range(NUM_CPU_NODES)]
    cluster.wait_for_nodes()

    yield {node.node_id for node in nodes}

    ray.shutdown()
    cluster.shutdown()


@pytest.fixture
def config(request: pytest.FixtureRequest) -> PolarsRayClusterConfig:
    return make_config(**getattr(request, "param", {}))


@pytest.fixture
def scheduler(
    cpu_node_ids: set[str],  # noqa: ARG001
    config: PolarsRayClusterConfig,
) -> typing.Iterator[ActorHandle]:
    yield from run_scheduler(config)


def test_one_worker_per_cpu_node(
    cpu_node_ids: set[str],
    config: PolarsRayClusterConfig,
    scheduler: ActorHandle,
) -> None:
    ray.get(scheduler.rescale_worker_pool_to.remote(NUM_CPU_NODES + 1))

    workers = wait_for_workers(config.cluster_id, NUM_CPU_NODES)

    node_ids = {worker: node_id(actor) for worker, actor in workers.items()}
    assert set(node_ids.values()) == cpu_node_ids
    for worker, node in node_ids.items():
        assert worker == WorkerSlot(node, 0).worker_id


def test_repeated_upscale_adds_no_extra_workers(
    config: PolarsRayClusterConfig,
    scheduler: ActorHandle,
) -> None:
    ray.get([scheduler.rescale_worker_pool_to.remote(NUM_CPU_NODES) for _ in range(3)])

    wait_for_workers(config.cluster_id, NUM_CPU_NODES)


@pytest.mark.parametrize("config", [{"workers_per_node": 2}], indirect=True)
def test_several_workers_per_node_spread_first(
    config: PolarsRayClusterConfig,
    scheduler: ActorHandle,
) -> None:
    ray.get(scheduler.rescale_worker_pool_to.remote(3))

    workers = wait_for_workers(config.cluster_id, 3)

    assert len({node_id(actor) for actor in workers.values()}) == NUM_CPU_NODES
    assert sorted(worker.index for worker in workers) == [0, 0, 1]


def test_worker_host_is_the_node_ip(
    config: PolarsRayClusterConfig,
    scheduler: ActorHandle,
) -> None:
    ray.get(scheduler.rescale_worker_pool_to.remote(1))
    (worker,) = wait_for_workers(config.cluster_id, 1).values()

    node_ips = {node["NodeID"]: node["NodeManagerAddress"] for node in ray.nodes()}
    host = ray.get(worker.__ray_call__.remote(lambda actor: actor.worker_host))

    assert host == node_ips[node_id(worker)]


def test_stop_without_dashboard(
    config: PolarsRayClusterConfig,
    scheduler: ActorHandle,
) -> None:
    ray.get(scheduler.rescale_worker_pool_to.remote(NUM_CPU_NODES))
    wait_for_workers(config.cluster_id, NUM_CPU_NODES)

    PolarsRayCluster(config).stop()

    assert list_workers(config.cluster_id) == set()


@pytest.mark.parametrize("config", [{"num_workers": NUM_CPU_NODES + 1}], indirect=True)
def test_start_fails_fast_without_room(
    cpu_node_ids: set[str],  # noqa: ARG001
    config: PolarsRayClusterConfig,
) -> None:
    with pytest.raises(RuntimeError, match="Cannot start 3 workers"):
        PolarsRayCluster(config).start()

    assert list_workers(config.cluster_id) == set()
    with pytest.raises(ValueError, match=SCHEDULER.actor_name):
        ray.get_actor(SCHEDULER.actor_name, namespace=config.cluster_id)


def _assert_stopped(config: PolarsRayClusterConfig) -> None:
    assert list_workers(config.cluster_id) == set()
    with pytest.raises(ValueError, match=SCHEDULER.actor_name):
        ray.get_actor(SCHEDULER.actor_name, namespace=config.cluster_id)


def test_start_waits_for_the_scheduler_and_workers(
    cpu_node_ids: set[str],  # noqa: ARG001
) -> None:
    scheduler = PolarsSchedulerConfig(
        cpus_hint=0,
        memory_hint=1,
        client_port=free_port(),
        worker_registration_port=free_port(),
        observatory=PolarsObservatoryConfig(enabled=False),
        scaling=PolarsScalingConfig(enabled=True, port=free_port()),
    )
    config = make_config(
        binary_path=str(LISTENING_ENGINE),
        num_workers=NUM_CPU_NODES,
        scheduler=scheduler,
    )
    cluster = PolarsRayCluster(config)

    try:
        cluster.start()

        assert len(list_workers(config.cluster_id)) == NUM_CPU_NODES
        assert ray.get(cluster._scheduler_actor.not_ready.remote()) == []
        host = ray.get(cluster._scheduler_actor.get_host.remote())
        assert cluster.get_client_addr() == f"{host}:{scheduler.client_port}"
        assert cluster.get_dashboard_addr() == (
            "The observatory is not enabled on this cluster"
        )
    finally:
        cluster.stop()

    _assert_stopped(config)


def test_start_fails_fast_when_a_worker_exits(
    cpu_node_ids: set[str],  # noqa: ARG001
) -> None:
    with socket.socket() as listener:
        listener.bind(("", 0))
        listener.listen()
        port = listener.getsockname()[1]
        config = make_config(
            binary_path=str(LISTENING_ENGINE),
            num_workers=1,
            worker=PolarsWorkerConfig(cpus_hint=1, memory_hint=1, task_port=port),
        )

        started = time.monotonic()
        with pytest.raises(WorkerExited, match=rf"{port} is already in use"):
            PolarsRayCluster(config).start()

    assert time.monotonic() - started < 15
    PolarsRayCluster(config).stop()
    _assert_stopped(config)


def test_start_fails_fast_when_the_scheduler_exits(
    cpu_node_ids: set[str],  # noqa: ARG001
) -> None:
    config = make_config(
        binary_path=str(FAILING_ENGINE), num_workers=0, scheduler_max_restarts=1
    )

    started = time.monotonic()
    with pytest.raises(RuntimeError, match="scheduler exited"):
        PolarsRayCluster(config).start()

    assert time.monotonic() - started < 15
    PolarsRayCluster(config).stop()
    _assert_stopped(config)


@ray.remote
class _Hog:
    """Other Ray work, holding CPUs of a node."""

    def ping(self) -> None:
        pass


@contextlib.contextmanager
def cpus_held(node: str, cpus: float) -> typing.Iterator[None]:
    hog = _Hog.options(  # type: ignore[attr-defined]
        num_cpus=cpus,
        scheduling_strategy=NodeAffinitySchedulingStrategy(node, soft=False),
    ).remote()
    ray.get(hog.ping.remote())
    try:
        yield
    finally:
        ray.kill(hog)
        _wait_for_free_cpus(node, cpus)


def _wait_for_free_cpus(node: str, cpus: float) -> None:
    deadline = time.monotonic() + 10
    while (
        available_resources_per_node().get(node, {}).get("CPU", 0) < cpus
        and time.monotonic() < deadline
    ):
        time.sleep(0.1)


@pytest.mark.parametrize(
    "config", [{"worker": PolarsWorkerConfig(cpus_hint=2)}], indirect=True
)
def test_workers_skip_a_node_whose_cpus_other_ray_work_holds(
    cpu_node_ids: set[str],
    config: PolarsRayClusterConfig,
    scheduler: ActorHandle,
) -> None:
    full, other = sorted(cpu_node_ids)
    with cpus_held(full, 1):
        ray.get(scheduler.rescale_worker_pool_to.remote(1))

        assert set(wait_for_workers(config.cluster_id, 1)) == {
            WorkerSlot(other, 0).worker_id
        }


@pytest.mark.parametrize(
    "config", [{"worker": PolarsWorkerConfig(cpus_hint=2)}], indirect=True
)
def test_a_worker_pinned_to_a_full_node_is_waiting(
    cpu_node_ids: set[str],
    config: PolarsRayClusterConfig,
) -> None:
    full = min(cpu_node_ids)
    workers = RayWorkers(config, scheduler_host="127.0.0.1")
    with cpus_held(full, 1):
        workers.start(WorkerSlot(full, 0))
        try:
            time.sleep(2)

            assert workers.waiting() == {WorkerSlot(full, 0).worker_id}
        finally:
            workers.terminate({WorkerSlot(full, 0).worker_id})


def test_start_names_the_workers_waiting_for_room_and_stops(
    cpu_node_ids: set[str],
) -> None:
    full = min(cpu_node_ids)
    config = make_config(
        binary_path=str(LISTENING_ENGINE),
        num_workers=1,
        startup_timeout=5,
        worker=PolarsWorkerConfig(cpus_hint=2, memory_hint=1),
    )
    # placed as a race with other Ray work would: on a node that has no room left
    workers = RayWorkers(config, scheduler_host="127.0.0.1")
    with cpus_held(full, 1):
        workers.start(WorkerSlot(full, 0))
        try:
            started = time.monotonic()
            with pytest.raises(
                RuntimeError,
                match=rf"{WorkerSlot(full, 0).worker_id} .*waiting for room",
            ):
                PolarsRayCluster(config).start()

            # stopping skips asking the waiting worker, without a process, to stop
            assert time.monotonic() - started < config.startup_timeout + 15

            with pytest.raises(ValueError, match=SCHEDULER.actor_name):
                ray.get_actor(SCHEDULER.actor_name, namespace=config.cluster_id)
        finally:
            workers.terminate({WorkerSlot(full, 0).worker_id})


@pytest.mark.parametrize(
    "config",
    [
        {
            "worker": PolarsWorkerConfig(cpus_hint=2, memory_hint=1),
            "num_workers": 1,
            "worker_reconcile_interval": 0.5,
            "startup_timeout": 2,
        }
    ],
    indirect=True,
)
def test_refill_moves_a_worker_waiting_for_room_to_a_node_with_room(
    cpu_node_ids: set[str],
    config: PolarsRayClusterConfig,
) -> None:
    full, other = sorted(cpu_node_ids)
    workers = RayWorkers(config, scheduler_host="127.0.0.1")
    with cpus_held(full, 1):
        workers.start(WorkerSlot(full, 0))
        try:
            scheduler = run_scheduler(config)
            next(scheduler)

            deadline = time.monotonic() + 30
            while (
                list_workers(config.cluster_id) != {WorkerSlot(other, 0).worker_id}
                and time.monotonic() < deadline
            ):
                time.sleep(0.5)

            assert list_workers(config.cluster_id) == {WorkerSlot(other, 0).worker_id}
            next(scheduler, None)
        finally:
            workers.terminate({WorkerSlot(full, 0).worker_id})
