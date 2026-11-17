"""The scheduler refills the worker pool on a simulated multi-node Ray cluster.

Uses its own Ray cluster, as it removes and adds nodes.
"""

import time
import typing

import pytest
import ray
from ray.actor import ActorHandle
from ray.cluster_utils import Cluster

from polars_nanook_ray.actors import (
    PolarsSchedulerActor,
    list_worker_names,
    resolve_scheduler_name,
    terminate_actors,
)
from polars_nanook_ray.cluster import PolarsRayCluster
from polars_nanook_ray.config import PolarsRayClusterConfig

from .test_worker_placement import _config, _node_id, _wait_for_workers


@pytest.fixture
def cluster() -> typing.Iterator[Cluster]:
    cluster = Cluster(
        initialize_head=True,
        head_node_args={
            "num_cpus": 0,
            "resources": {"head": 1},
            "include_dashboard": False,
        },
    )
    ray.init(address=cluster.address)

    yield cluster

    ray.shutdown()
    cluster.shutdown()


@pytest.fixture
def config() -> PolarsRayClusterConfig:
    return _config(num_workers=2, worker_reconcile_interval=0.5)


@pytest.fixture
def scheduler(
    cluster: Cluster, config: PolarsRayClusterConfig
) -> typing.Iterator[ActorHandle]:
    cluster.add_node(num_cpus=2)
    cluster.add_node(num_cpus=2)
    cluster.wait_for_nodes()

    actor = PolarsSchedulerActor.options(  # type: ignore[attr-defined]
        name=resolve_scheduler_name(),
        namespace=config.cluster_id,
        resources={"head": 0.001},
        num_cpus=0,
    ).remote(config)

    yield actor

    PolarsRayCluster(config).stop()
    terminate_actors([actor], config.actor_response_timeout)


def test_scheduler_fills_the_pool_to_num_workers(
    config: PolarsRayClusterConfig, scheduler: ActorHandle
) -> None:
    ray.get(scheduler.get_host.remote())

    _wait_for_workers(config.cluster_id, 2)


def test_worker_past_its_restart_limit_is_replaced(
    config: PolarsRayClusterConfig, scheduler: ActorHandle
) -> None:
    ray.get(scheduler.get_host.remote())
    workers = _wait_for_workers(config.cluster_id, 2)

    name, actor = next(iter(workers.items()))
    ray.kill(actor, no_restart=True)

    workers = _wait_for_workers(config.cluster_id, 2)
    assert name in workers


def test_worker_on_a_lost_node_moves_to_a_new_node(
    cluster: Cluster, config: PolarsRayClusterConfig, scheduler: ActorHandle
) -> None:
    ray.get(scheduler.get_host.remote())
    workers = _wait_for_workers(config.cluster_id, 2)
    lost = next(
        node
        for node in cluster.list_all_nodes()
        if node.node_id in {_node_id(actor) for actor in workers.values()}
    )

    cluster.remove_node(lost)
    new = cluster.add_node(num_cpus=2)
    cluster.wait_for_nodes()

    workers = _wait_for_workers(config.cluster_id, 2)
    node_ids = {_node_id(actor) for actor in workers.values()}
    assert new.node_id in node_ids
    assert lost.node_id not in node_ids


def test_stop_leaves_no_workers_behind(
    config: PolarsRayClusterConfig, scheduler: ActorHandle
) -> None:
    ray.get(scheduler.get_host.remote())
    _wait_for_workers(config.cluster_id, 2)

    PolarsRayCluster(config).stop()
    time.sleep(3 * config.worker_reconcile_interval)

    assert list_worker_names(config.cluster_id) == set()
