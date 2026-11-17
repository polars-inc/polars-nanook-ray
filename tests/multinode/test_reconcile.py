"""The scheduler refills the worker pool on a simulated multi-node Ray cluster.

Uses its own Ray cluster, as it removes and adds nodes.
"""

import os
import time
import typing

import pytest
import ray
from ray.actor import ActorHandle
from ray.cluster_utils import Cluster

from polars_nanook_ray.cluster import PolarsRayCluster
from polars_nanook_ray.config import PolarsRayClusterConfig
from polars_nanook_ray.identity import list_workers

from .utils import head_cluster, make_config, node_id, run_scheduler, wait_for_workers


@pytest.fixture
def cluster() -> typing.Iterator[Cluster]:
    cluster = head_cluster()

    yield cluster

    ray.shutdown()
    cluster.shutdown()


@pytest.fixture
def config() -> PolarsRayClusterConfig:
    return make_config(num_workers=2, worker_reconcile_interval=0.5)


@pytest.fixture
def scheduler(
    cluster: Cluster, config: PolarsRayClusterConfig
) -> typing.Iterator[ActorHandle]:
    cluster.add_node(num_cpus=2)
    cluster.add_node(num_cpus=2)
    cluster.wait_for_nodes()

    yield from run_scheduler(config)


def test_scheduler_fills_the_pool_to_num_workers(
    config: PolarsRayClusterConfig, scheduler: ActorHandle
) -> None:
    ray.get(scheduler.get_host.remote())

    wait_for_workers(config.cluster_id, 2)


def test_worker_past_its_restart_limit_is_replaced(
    config: PolarsRayClusterConfig, scheduler: ActorHandle
) -> None:
    ray.get(scheduler.get_host.remote())
    workers = wait_for_workers(config.cluster_id, 2)

    worker, actor = next(iter(workers.items()))
    ray.kill(actor, no_restart=True)

    workers = wait_for_workers(config.cluster_id, 2)
    assert worker in workers


def test_worker_on_a_lost_node_moves_to_a_new_node(
    cluster: Cluster, config: PolarsRayClusterConfig, scheduler: ActorHandle
) -> None:
    ray.get(scheduler.get_host.remote())
    workers = wait_for_workers(config.cluster_id, 2)
    lost = next(
        node
        for node in cluster.list_all_nodes()
        if node.node_id in {node_id(actor) for actor in workers.values()}
    )

    cluster.remove_node(lost)
    new = cluster.add_node(num_cpus=2)
    cluster.wait_for_nodes()

    workers = wait_for_workers(config.cluster_id, 2)
    node_ids = {node_id(actor) for actor in workers.values()}
    assert new.node_id in node_ids
    assert lost.node_id not in node_ids


def test_stop_leaves_no_workers_behind(
    config: PolarsRayClusterConfig, scheduler: ActorHandle
) -> None:
    ray.get(scheduler.get_host.remote())
    wait_for_workers(config.cluster_id, 2)

    PolarsRayCluster(config).stop()
    time.sleep(3 * config.worker_reconcile_interval)

    assert list_workers(config.cluster_id) == set()


def _engine_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def test_workers_die_with_their_scheduler_and_its_restart_starts_new_ones(
    cluster: Cluster, config: PolarsRayClusterConfig
) -> None:
    cluster.add_node(num_cpus=2)
    cluster.add_node(num_cpus=2)
    cluster.wait_for_nodes()
    scheduler = run_scheduler(config, max_restarts=1)
    actor = next(scheduler)
    try:
        ray.get(actor.get_host.remote())
        workers = wait_for_workers(config.cluster_id, 2)
        old_engines = {
            ray.get(worker.get_worker_pid.remote()) for worker in workers.values()
        }

        ray.kill(actor, no_restart=False)

        deadline = time.monotonic() + 30
        while any(_engine_alive(pid) for pid in old_engines) and (
            time.monotonic() < deadline
        ):
            time.sleep(0.5)
        assert not any(_engine_alive(pid) for pid in old_engines)

        deadline = time.monotonic() + 30
        while True:  # until Ray has restarted the scheduler
            try:
                ray.get(actor.get_host.remote())
                break
            except ray.exceptions.ActorUnavailableError:
                assert time.monotonic() < deadline
                time.sleep(0.5)
        workers = wait_for_workers(config.cluster_id, 2)
        new_engines = {
            ray.get(worker.get_worker_pid.remote()) for worker in workers.values()
        }
        assert new_engines.isdisjoint(old_engines)
    finally:
        next(scheduler, None)
