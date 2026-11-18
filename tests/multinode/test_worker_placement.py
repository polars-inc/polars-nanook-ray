"""Worker placement on a simulated multi-node Ray cluster.

Several raylets run on this one machine, without the dashboard. The binaries are
replaced by a stub that sleeps, so no license is needed: only the actors are under test.
"""

import socket
import time
import typing

import pytest
import ray
from ray.actor import ActorHandle
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from polars_nanook_ray import Autoscale, Ports, RayClusterContext
from polars_nanook_ray.actors import WorkerExited
from polars_nanook_ray.identity import SCHEDULER, WorkerId, list_workers

from .utils import (
    FAILING_ENGINE,
    LISTENING_ENGINE,
    free_port,
    head_cluster,
    make_context,
    name_of,
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
def ctx(request: pytest.FixtureRequest) -> RayClusterContext:
    return make_context(**getattr(request, "param", {}))


@pytest.fixture
def scheduler(
    cpu_node_ids: set[str],  # noqa: ARG001
    ctx: RayClusterContext,
) -> typing.Iterator[ActorHandle]:
    yield from run_scheduler(ctx)


def test_workers_spread_over_the_nodes_with_room(
    cpu_node_ids: set[str],
    ctx: RayClusterContext,
    scheduler: ActorHandle,
) -> None:
    ray.get(scheduler.rescale_worker_pool_to.remote(NUM_CPU_NODES))

    workers = wait_for_workers(name_of(ctx), NUM_CPU_NODES)

    assert set(workers) == {WorkerId(i) for i in range(NUM_CPU_NODES)}
    assert {node_id(actor) for actor in workers.values()} == cpu_node_ids


def test_repeated_upscale_adds_no_extra_workers(
    ctx: RayClusterContext,
    scheduler: ActorHandle,
) -> None:
    ray.get([scheduler.rescale_worker_pool_to.remote(NUM_CPU_NODES) for _ in range(3)])

    wait_for_workers(name_of(ctx), NUM_CPU_NODES)


@pytest.mark.parametrize("ctx", [{"cpus": 2}], indirect=True)
def test_a_worker_waiting_for_room_exists_and_is_not_ready(
    ctx: RayClusterContext,
    scheduler: ActorHandle,
) -> None:
    ray.get(scheduler.rescale_worker_pool_to.remote(NUM_CPU_NODES + 1))

    workers = wait_for_workers(name_of(ctx), NUM_CPU_NODES + 1)

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
) -> None:
    ctx = make_context(binary_path=str(LISTENING_ENGINE), workers=2, startup_timeout=20)

    try:
        ctx.start()

        workers = wait_for_workers(name_of(ctx), 2)
        assert {node_id(actor) for actor in workers.values()} == {busy_node}
    finally:
        ctx.stop()


def test_worker_host_is_the_node_ip(
    ctx: RayClusterContext,
    scheduler: ActorHandle,
) -> None:
    ray.get(scheduler.rescale_worker_pool_to.remote(1))
    (worker,) = wait_for_workers(name_of(ctx), 1).values()

    node_ips = {node["NodeID"]: node["NodeManagerAddress"] for node in ray.nodes()}
    host = ray.get(worker.__ray_call__.remote(lambda actor: actor.worker_host))

    assert host == node_ips[node_id(worker)]


def test_stop_without_dashboard(
    ctx: RayClusterContext,
    scheduler: ActorHandle,
) -> None:
    ray.get(scheduler.rescale_worker_pool_to.remote(NUM_CPU_NODES))
    wait_for_workers(name_of(ctx), NUM_CPU_NODES)

    ctx.stop()

    assert list_workers(name_of(ctx)) == set()


@pytest.mark.parametrize("ctx", [{"workers": 1, "cpus": 4}], indirect=True)
def test_start_fails_fast_when_a_worker_fits_no_node(
    cpu_node_ids: set[str],  # noqa: ARG001
    ctx: RayClusterContext,
) -> None:
    started = time.monotonic()
    with pytest.raises(
        RuntimeError, match="requests 4 CPU, but the largest node has 2 CPU"
    ):
        ctx.start()

    assert time.monotonic() - started < 15

    assert list_workers(name_of(ctx)) == set()
    with pytest.raises(ValueError, match=SCHEDULER.actor_name):
        ray.get_actor(SCHEDULER.actor_name, namespace=name_of(ctx))


def test_start_times_out_on_workers_waiting_for_room(
    cpu_node_ids: set[str],  # noqa: ARG001
) -> None:
    ctx = make_context(
        binary_path=str(LISTENING_ENGINE),
        workers=NUM_CPU_NODES + 1,
        cpus=2,
        startup_timeout=5,
    )

    try:
        with pytest.raises(RuntimeError, match="waiting for a Ray node with 2 CPU"):
            ctx.start()
    finally:
        ctx.stop()

    _assert_stopped(ctx)


def _assert_stopped(ctx: RayClusterContext) -> None:
    assert list_workers(name_of(ctx)) == set()
    with pytest.raises(ValueError, match=SCHEDULER.actor_name):
        ray.get_actor(SCHEDULER.actor_name, namespace=name_of(ctx))


def test_start_waits_for_the_scheduler_and_workers(
    cpu_node_ids: set[str],  # noqa: ARG001
) -> None:
    client_port = free_port()
    ctx = make_context(
        binary_path=str(LISTENING_ENGINE),
        workers=NUM_CPU_NODES,
        autoscale=Autoscale(),
        ports=Ports(client=client_port, dashboard=None),
    )

    try:
        ctx.start()

        assert len(list_workers(name_of(ctx))) == NUM_CPU_NODES
        assert ray.get(ctx._scheduler_actor.not_ready.remote()) == []
        host = ray.get(ctx._scheduler_actor.get_host.remote())
        assert ctx.client_addr == f"http://{host}:{client_port}"
        assert ctx.dashboard_addr is None
    finally:
        ctx.stop()

    _assert_stopped(ctx)


def test_start_fails_fast_when_a_worker_exits(
    cpu_node_ids: set[str],  # noqa: ARG001
) -> None:
    with socket.socket() as listener:
        listener.bind(("", 0))
        listener.listen()
        port = listener.getsockname()[1]
        ctx = make_context(
            binary_path=str(LISTENING_ENGINE),
            workers=1,
            ports=Ports(client=None, dashboard=None, task=port),
        )

        started = time.monotonic()
        with pytest.raises(WorkerExited, match=rf"{port} is already in use"):
            ctx.start()

    assert time.monotonic() - started < 15
    ctx.stop()
    _assert_stopped(ctx)


def test_start_fails_fast_when_the_scheduler_exits(
    cpu_node_ids: set[str],  # noqa: ARG001
) -> None:
    ctx = make_context(
        binary_path=str(FAILING_ENGINE), workers=0, scheduler_max_restarts=1
    )

    started = time.monotonic()
    with pytest.raises(RuntimeError, match="scheduler engine exited with code 1"):
        ctx.start()

    assert time.monotonic() - started < 15
    ctx.stop()
    _assert_stopped(ctx)
