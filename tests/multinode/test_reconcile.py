"""The worker pool is kept at its target count on a simulated multi-node Ray cluster.

Uses its own Ray cluster, as it removes and adds nodes.
"""

import os
import time
import typing

import pytest
import ray
from ray.actor import ActorHandle
from ray.cluster_utils import Cluster

from polars_nanook_ray import RayClusterContext
from polars_nanook_ray.identity import list_workers

from .utils import (
    head_cluster,
    make_context,
    name_of,
    node_id,
    run_scheduler,
    wait_for_workers,
)


@pytest.fixture
def cluster() -> typing.Iterator[Cluster]:
    cluster = head_cluster()

    yield cluster

    ray.shutdown()
    cluster.shutdown()


@pytest.fixture
def ctx() -> RayClusterContext:
    return make_context(workers=2, reconcile_interval=0.5)


@pytest.fixture
def scheduler(cluster: Cluster, ctx: RayClusterContext) -> typing.Iterator[ActorHandle]:
    cluster.add_node(num_cpus=2)
    cluster.add_node(num_cpus=2)
    cluster.wait_for_nodes()

    yield from run_scheduler(ctx)


def test_scheduler_fills_the_pool_to_num_workers(
    ctx: RayClusterContext, scheduler: ActorHandle
) -> None:
    ray.get(scheduler.get_host.remote())

    wait_for_workers(name_of(ctx), 2)


def test_worker_past_its_restart_limit_is_replaced(
    ctx: RayClusterContext, scheduler: ActorHandle
) -> None:
    ray.get(scheduler.get_host.remote())
    workers = wait_for_workers(name_of(ctx), 2)

    worker, actor = next(iter(workers.items()))
    ray.kill(actor, no_restart=True)

    workers = wait_for_workers(name_of(ctx), 2)
    assert worker in workers


def test_ray_restarts_a_worker_of_a_lost_node_on_another_node(
    cluster: Cluster,
) -> None:
    cluster.add_node(num_cpus=2)
    cluster.wait_for_nodes()
    # no reconciling: only Ray's restart can bring the worker back
    ctx = make_context(workers=1, reconcile_interval=0)
    scheduler = run_scheduler(ctx)
    actor = next(scheduler)
    try:
        ray.get(actor.ensure_workers.remote(1))
        ((worker, handle),) = wait_for_workers(name_of(ctx), 1).items()
        lost = next(
            node for node in cluster.list_all_nodes() if node.node_id == node_id(handle)
        )

        new = cluster.add_node(num_cpus=2)
        cluster.remove_node(lost)
        cluster.wait_for_nodes()

        deadline = time.monotonic() + 60
        while True:
            try:
                assert node_id(handle) == new.node_id
                break
            except ray.exceptions.ActorUnavailableError:
                assert time.monotonic() < deadline
                time.sleep(0.5)
        assert set(wait_for_workers(name_of(ctx), 1)) == {worker}
    finally:
        next(scheduler, None)


def test_stop_leaves_no_workers_behind(
    ctx: RayClusterContext, scheduler: ActorHandle
) -> None:
    ray.get(scheduler.get_host.remote())
    wait_for_workers(name_of(ctx), 2)

    ctx.stop()
    time.sleep(3 * ctx._spec.reconcile_interval)

    assert list_workers(name_of(ctx)) == set()


def _engine_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def test_workers_die_with_their_scheduler_and_its_restart_starts_new_ones(
    cluster: Cluster, ctx: RayClusterContext
) -> None:
    cluster.add_node(num_cpus=2)
    cluster.add_node(num_cpus=2)
    cluster.wait_for_nodes()
    scheduler = run_scheduler(ctx, max_restarts=1)
    actor = next(scheduler)
    try:
        ray.get(actor.get_host.remote())
        workers = wait_for_workers(name_of(ctx), 2)
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
        workers = wait_for_workers(name_of(ctx), 2)
        new_engines = {
            ray.get(worker.get_worker_pid.remote()) for worker in workers.values()
        }
        assert new_engines.isdisjoint(old_engines)
    finally:
        next(scheduler, None)
