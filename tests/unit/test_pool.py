import threading
import time
import typing

import pytest

from polars_nanook_ray.actors.pool import (
    Node,
    NoRoomForWorkers,
    WorkerExited,
    WorkerPool,
)
from polars_nanook_ray.config import (
    PolarsEnterpriseLicenseConfig,
    PolarsRayClusterConfig,
    PolarsSchedulerConfig,
    PolarsWorkerConfig,
)
from polars_nanook_ray.identity import WorkerId

GIB = 1024**3


def workers_named(*indices: int) -> set[WorkerId]:
    return {WorkerId(index) for index in indices}


class InMemoryWorkers:
    """Named worker actors, as Ray holds them, on nodes Ray places them on.

    A started worker is listed at once, as a named actor is, even while it waits for
    room on a node; it is ready once it listens.
    """

    def __init__(self, *nodes: Node) -> None:
        self.alive_nodes = list(nodes) or [Node(cpus=64, memory=256 * GIB)]
        self.workers: set[WorkerId] = set()
        self.listening: set[WorkerId] = set()
        self.exited: set[WorkerId] = set()  # past their restart limit

    def nodes(self) -> list[Node]:
        return self.alive_nodes

    def running(self) -> set[WorkerId]:
        return set(self.workers)

    def start(self, worker: WorkerId) -> None:
        self.workers.add(worker)

    def terminate(self, workers: typing.Collection[WorkerId]) -> None:
        self.workers.difference_update(workers)

    def ready(self, workers: typing.Collection[WorkerId]) -> set[WorkerId]:
        if exited := sorted(self.exited.intersection(workers)):
            msg = f"{exited[0]} exited while starting"
            raise WorkerExited(msg)
        return self.listening.intersection(workers)

    def kill(self, worker: WorkerId) -> None:
        """Remove `worker` as Ray does once it is past its restart limit."""
        self.workers.remove(worker)


def config(
    cpus: int = 1,
    memory: int = 1,
    reconcile_interval: float = 10,
    **kwargs: typing.Any,
) -> PolarsRayClusterConfig:
    """Return a cluster config, with `memory` in GiB."""
    return PolarsRayClusterConfig(
        license=PolarsEnterpriseLicenseConfig(license_path="/nonexistent.json"),
        scheduler=PolarsSchedulerConfig(),
        worker=PolarsWorkerConfig(cpus_hint=cpus, memory_hint=memory * GIB),
        worker_reconcile_interval=reconcile_interval,
        **kwargs,
    )


def pool(workers: InMemoryWorkers, target: int = 0, **kwargs: typing.Any) -> WorkerPool:
    return WorkerPool(workers, config(**kwargs), target)


def test_ensure_starts_workers_named_from_zero() -> None:
    workers = InMemoryWorkers()

    assert pool(workers).ensure(3) == workers_named(0, 1, 2)
    assert workers.running() == workers_named(0, 1, 2)


def test_ensure_names_new_workers_after_the_lowest_free_indices() -> None:
    workers = InMemoryWorkers()
    workers.workers = workers_named(1, 3)

    assert pool(workers).ensure(4) == workers_named(0, 1, 2, 3)


def test_ensure_never_adds_to_or_removes_from_a_larger_pool() -> None:
    workers = InMemoryWorkers()
    worker_pool = pool(workers)
    worker_pool.ensure(3)

    assert worker_pool.ensure(1) == workers_named(0, 1, 2)


def test_ensure_fails_fast_when_a_worker_fits_no_node() -> None:
    workers = InMemoryWorkers(
        Node(cpus=2, memory=8 * GIB), Node(cpus=4, memory=8 * GIB)
    )
    worker_pool = pool(workers, target=1, cpus=8)

    with pytest.raises(
        NoRoomForWorkers, match="requests 8 CPU, but the largest node has 4 CPU"
    ):
        worker_pool.ensure(2)

    assert workers.running() == set()
    assert worker_pool.status().target == 1


def test_ensure_names_the_memory_no_node_has() -> None:
    workers = InMemoryWorkers(Node(cpus=8, memory=4 * GIB))

    with pytest.raises(
        NoRoomForWorkers, match=r"requests 16 GiB of memory, but .* has 4 GiB"
    ):
        pool(workers, memory=16).ensure(1)


def test_ensure_fails_when_no_single_node_has_both() -> None:
    workers = InMemoryWorkers(
        Node(cpus=8, memory=1 * GIB), Node(cpus=1, memory=8 * GIB)
    )

    with pytest.raises(NoRoomForWorkers, match="no single node has both"):
        pool(workers, cpus=4, memory=4).ensure(1)


def test_ensure_starts_workers_that_fit_a_node_even_if_it_is_busy() -> None:
    workers = InMemoryWorkers(Node(cpus=2, memory=4 * GIB))

    assert pool(workers, cpus=2, memory=4).ensure(3) == workers_named(0, 1, 2)


def test_ensure_checks_no_room_when_enough_workers_exist() -> None:
    workers = InMemoryWorkers(Node(cpus=1, memory=GIB))
    workers.workers = workers_named(0)

    assert pool(workers, cpus=8).ensure(1) == workers_named(0)


def test_a_worker_waiting_for_room_counts_as_existing() -> None:
    workers = InMemoryWorkers()  # none of the workers listen yet
    worker_pool = pool(workers)
    worker_pool.ensure(2)

    worker_pool.ensure(2)
    worker_pool.rescale(2)
    worker_pool.reconcile()

    assert workers.running() == workers_named(0, 1)


def test_rescale_up_starts_the_missing_workers() -> None:
    workers = InMemoryWorkers()
    worker_pool = pool(workers)

    worker_pool.rescale(3)

    assert workers.running() == workers_named(0, 1, 2)
    assert worker_pool.status().target == 3


def test_rescale_up_starts_workers_no_node_fits_yet() -> None:
    # left pending, for Ray's autoscaler to add a larger node
    workers = InMemoryWorkers(Node(cpus=1, memory=GIB))

    pool(workers, cpus=8).rescale(1)

    assert workers.running() == workers_named(0)


def test_repeated_rescale_up_adds_no_extra_workers() -> None:
    workers = InMemoryWorkers()
    worker_pool = pool(workers)

    for _ in range(3):
        worker_pool.rescale(2)

    assert workers.running() == workers_named(0, 1)


@pytest.mark.parametrize(("requested", "clamped"), [(0, 1), (1, 1), (2, 2), (3, 2)])
def test_rescale_is_clamped_to_min_and_max_workers(
    requested: int, clamped: int
) -> None:
    workers = InMemoryWorkers()
    worker_pool = pool(workers, min_workers=1, max_workers=2)

    worker_pool.rescale(requested)

    assert worker_pool.status() == (workers.running(), clamped)
    assert len(workers.running()) == clamped


def test_rescale_to_zero_removes_every_worker() -> None:
    workers = InMemoryWorkers()
    worker_pool = pool(workers)
    worker_pool.ensure(2)

    worker_pool.rescale(0, keep={"worker-0"})

    assert workers.running() == set()


def test_rescale_down_removes_the_highest_indices_first() -> None:
    workers = InMemoryWorkers()
    workers.workers = workers_named(0, 2, 5, 7)
    worker_pool = pool(workers)

    worker_pool.rescale(2)

    assert workers.running() == workers_named(0, 2)


def test_rescale_down_deletes_the_workers_named() -> None:
    workers = InMemoryWorkers()
    worker_pool = pool(workers)
    worker_pool.ensure(3)

    # an engine instance ID, an actor name, and a name that is no worker
    worker_pool.rescale(1, delete={"worker-0-0123abcd", "worker-2", "nope"})

    assert workers.running() == workers_named(1)


def test_rescale_down_keeps_the_workers_named() -> None:
    workers = InMemoryWorkers()
    worker_pool = pool(workers)
    worker_pool.ensure(3)

    worker_pool.rescale(1, keep={"worker-1-0123abcd", "nope"})

    assert workers.running() == workers_named(1)


def test_rescale_down_deletes_then_keeps() -> None:
    workers = InMemoryWorkers()
    worker_pool = pool(workers)
    worker_pool.ensure(3)

    worker_pool.rescale(1, delete={"worker-0"}, keep={"worker-0", "worker-1"})

    assert workers.running() == workers_named(1)


def test_rescale_down_ignores_names_that_are_no_workers() -> None:
    workers = InMemoryWorkers()
    worker_pool = pool(workers)
    worker_pool.ensure(3)

    worker_pool.rescale(2, keep={"nope"})

    assert workers.running() == workers_named(0, 1)


def test_an_upscale_after_a_delete_reuses_the_freed_index() -> None:
    workers = InMemoryWorkers()
    worker_pool = pool(workers)
    worker_pool.ensure(3)
    worker_pool.rescale(2, delete={"worker-0"})

    worker_pool.rescale(3)

    assert workers.running() == workers_named(0, 1, 2)


def test_reconcile_replaces_a_worker_past_its_restart_limit() -> None:
    workers = InMemoryWorkers()
    worker_pool = pool(workers)
    worker_pool.ensure(3)
    workers.kill(WorkerId(1))

    assert worker_pool.reconcile() == [WorkerId(1)]
    assert workers.running() == workers_named(0, 1, 2)


def test_reconcile_does_nothing_at_the_target() -> None:
    workers = InMemoryWorkers()
    worker_pool = pool(workers)
    worker_pool.ensure(2)

    assert worker_pool.reconcile() == []


class SlowStart(InMemoryWorkers):
    """Workers whose start blocks until released, as a slow Ray call would."""

    def __init__(self) -> None:
        super().__init__()
        self.starting = threading.Event()
        self.release = threading.Event()

    def start(self, worker: WorkerId) -> None:
        self.starting.set()
        self.release.wait(5)
        super().start(worker)


def test_a_downscale_during_a_reconcile_is_not_undone() -> None:
    workers = SlowStart()
    workers.workers = workers_named(0)
    worker_pool = pool(workers, target=2)

    reconcile = threading.Thread(target=worker_pool.reconcile)
    reconcile.start()
    workers.starting.wait(5)
    rescale = threading.Thread(target=worker_pool.rescale, args=(1,))
    rescale.start()
    time.sleep(0.1)  # for the rescale to wait on the reconcile
    workers.release.set()
    reconcile.join(5)
    rescale.join(5)
    worker_pool.reconcile()

    assert len(workers.running()) == 1


def wait_for(condition: typing.Callable[[], bool]) -> bool:
    deadline = time.monotonic() + 5
    while not condition() and time.monotonic() < deadline:
        time.sleep(0.01)
    return condition()


def test_reconciling_runs_in_the_background_until_halted() -> None:
    workers = InMemoryWorkers()
    worker_pool = WorkerPool(workers, config(reconcile_interval=0.01), 2)

    worker_pool.start_reconciling()
    assert wait_for(lambda: len(workers.running()) == 2)

    worker_pool.halt()
    workers.kill(WorkerId(0))
    time.sleep(0.1)

    assert workers.running() == workers_named(1)


def test_no_reconciling_at_a_zero_interval() -> None:
    workers = InMemoryWorkers()
    worker_pool = WorkerPool(workers, config(reconcile_interval=0), 1)

    worker_pool.start_reconciling()
    time.sleep(0.1)

    assert workers.running() == set()


def test_stop_removes_every_worker_for_good() -> None:
    workers = InMemoryWorkers()
    worker_pool = pool(workers)
    worker_pool.ensure(2)

    worker_pool.stop()
    worker_pool.reconcile()

    assert workers.running() == set()
    assert worker_pool.status().target == 0


def test_ensure_raises_a_lower_target() -> None:
    workers = InMemoryWorkers()
    worker_pool = pool(workers, target=0)

    worker_pool.ensure(2)

    assert worker_pool.status().target == 2


class SlowListing(InMemoryWorkers):
    """Workers listed only once they are up, should Ray list a new actor late."""

    def start(self, worker: WorkerId) -> None:
        pass


def test_ensure_returns_the_workers_it_started_before_they_are_listed() -> None:
    workers = SlowListing()
    workers.workers = workers_named(0)

    assert pool(workers).ensure(2) == workers_named(0, 1)


def test_not_ready_names_the_workers_not_listening_yet() -> None:
    workers = InMemoryWorkers()
    worker_pool = pool(workers)
    worker_pool.ensure(2)

    workers.listening.add(WorkerId(0))
    assert worker_pool.not_ready() == workers_named(1)

    workers.listening.add(WorkerId(1))
    assert worker_pool.not_ready() == set()


def test_not_ready_awaits_the_workers_started_before_they_are_listed() -> None:
    workers = SlowListing()
    workers.workers = workers_named(0)
    workers.listening.add(WorkerId(0))
    worker_pool = pool(workers)

    worker_pool.ensure(2)

    assert worker_pool.not_ready() == workers_named(1)


def test_not_ready_raises_when_a_started_worker_exited() -> None:
    workers = InMemoryWorkers()
    worker_pool = pool(workers)
    worker_pool.ensure(1)

    workers.kill(WorkerId(0))
    workers.exited.add(WorkerId(0))

    with pytest.raises(WorkerExited, match="worker-0"):
        worker_pool.not_ready()


def test_not_ready_leaves_a_worker_seen_ready_to_the_reconcile() -> None:
    workers = InMemoryWorkers()
    worker_pool = pool(workers)
    worker_pool.ensure(1)
    workers.listening.add(WorkerId(0))
    worker_pool.not_ready()

    workers.kill(WorkerId(0))
    workers.exited.add(WorkerId(0))

    assert worker_pool.not_ready() == set()


def test_not_ready_forgets_the_workers_stopped() -> None:
    workers = SlowListing()
    worker_pool = pool(workers)
    worker_pool.ensure(1)

    worker_pool.stop()

    assert worker_pool.not_ready() == set()
