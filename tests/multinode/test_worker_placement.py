"""Worker placement on a simulated multi-node Ray cluster.

Several raylets run on this one machine, without the dashboard. The binaries are
replaced by `true`, so no license is needed: only the actors are under test.
"""

import time
import typing

import pytest
import ray
from ray.actor import ActorHandle
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from polars_nanook_ray.actors import list_worker_names, worker_name
from polars_nanook_ray.cluster import PolarsRayCluster
from polars_nanook_ray.config import PolarsRayClusterConfig

from .utils import head_cluster, make_config, node_id, run_scheduler, wait_for_workers

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


def test_workers_spread_over_the_nodes_with_room(
    cpu_node_ids: set[str],
    config: PolarsRayClusterConfig,
    scheduler: ActorHandle,
) -> None:
    ray.get(scheduler.rescale_worker_pool_to.remote(NUM_CPU_NODES))

    workers = wait_for_workers(config.cluster_id, NUM_CPU_NODES)

    assert set(workers) == {worker_name(i) for i in range(NUM_CPU_NODES)}
    assert {node_id(actor) for actor in workers.values()} == cpu_node_ids


def test_repeated_upscale_adds_no_extra_workers(
    config: PolarsRayClusterConfig,
    scheduler: ActorHandle,
) -> None:
    ray.get([scheduler.rescale_worker_pool_to.remote(NUM_CPU_NODES) for _ in range(3)])

    wait_for_workers(config.cluster_id, NUM_CPU_NODES)


@pytest.mark.parametrize("config", [{"worker_cpus": 2}], indirect=True)
def test_a_worker_waiting_for_room_exists_and_is_not_placed(
    config: PolarsRayClusterConfig,
    scheduler: ActorHandle,
) -> None:
    ray.get(scheduler.rescale_worker_pool_to.remote(NUM_CPU_NODES + 1))

    workers = wait_for_workers(config.cluster_id, NUM_CPU_NODES + 1)

    placed = ray.wait(
        [actor.__ray_ready__.remote() for actor in workers.values()],
        num_returns=NUM_CPU_NODES + 1,
        timeout=10,
    )[0]
    assert len(placed) == NUM_CPU_NODES


@pytest.fixture
def busy_node(cpu_node_ids: set[str]) -> typing.Iterator[str]:
    """Fill one CPU node with other Ray work; yield the other one."""
    busy, free = sorted(cpu_node_ids)
    blocker = Blocker.options(  # type: ignore[attr-defined]
        num_cpus=2,
        scheduling_strategy=NodeAffinitySchedulingStrategy(busy, soft=False),
    ).remote()
    ray.get(blocker.__ray_ready__.remote())

    yield free

    ray.kill(blocker)


@ray.remote
class Blocker:
    pass


def test_workers_go_to_the_node_with_room_when_another_is_busy(
    busy_node: str,
    config: PolarsRayClusterConfig,
    scheduler: ActorHandle,
) -> None:
    ray.get(scheduler.rescale_worker_pool_to.remote(2))

    workers = wait_for_workers(config.cluster_id, 2)

    placed = ray.wait(
        [actor.__ray_ready__.remote() for actor in workers.values()],
        num_returns=2,
        timeout=10,
    )[0]
    assert len(placed) == 2
    assert {node_id(actor) for actor in workers.values()} == {busy_node}


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

    assert list_worker_names(config.cluster_id) == set()


@pytest.mark.parametrize(
    "config", [{"num_workers": 1, "worker_cpus": 4}], indirect=True
)
def test_start_fails_fast_when_a_worker_fits_no_node(
    cpu_node_ids: set[str],  # noqa: ARG001
    config: PolarsRayClusterConfig,
) -> None:
    started = time.monotonic()
    with pytest.raises(
        RuntimeError, match="requests 4 CPU, but the largest node has 2 CPU"
    ):
        PolarsRayCluster(config).start()

    assert time.monotonic() - started < 15
    assert list_worker_names(config.cluster_id) == set()
