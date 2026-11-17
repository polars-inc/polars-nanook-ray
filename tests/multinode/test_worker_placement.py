"""Worker placement on a simulated multi-node Ray cluster.

Several raylets run on this one machine, without the dashboard. The binaries are
replaced by a stub that sleeps, so no license is needed: only the actors are under test.
"""

import typing

import pytest
import ray
from ray.actor import ActorHandle

from polars_nanook_ray.actors import WorkerSlot
from polars_nanook_ray.cluster import PolarsRayCluster
from polars_nanook_ray.config import PolarsRayClusterConfig
from polars_nanook_ray.identity import list_workers

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
