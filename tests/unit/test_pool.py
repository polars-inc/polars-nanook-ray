import dataclasses
import itertools
import threading
import time
import typing

import pytest

from polars_nanook_ray.actors.pool import (
    MAX_REFILL_BACKOFF,
    Node,
    NoRoomForWorkers,
    WorkerExited,
    WorkerPool,
    WorkerSlot,
)
from polars_nanook_ray.config import (
    PolarsEnterpriseLicenseConfig,
    PolarsRayClusterConfig,
    PolarsSchedulerConfig,
    PolarsWorkerConfig,
)
from polars_nanook_ray.identity import WorkerId

NODE_A = "a" * 56
NODE_B = "b" * 56
NODE_C = "c" * 56


def slot(node_id: str, index: int = 0) -> WorkerId:
    return WorkerSlot(node_id, index).worker_id


class InMemoryWorkers:
    """Nodes and named worker actors, as Ray would hold them.

    A worker started on a node without room for its `cpus_hint` and `memory_hint` waits
    for it, as Ray holds a pinned actor pending until its node has room.
    """

    def __init__(
        self, *nodes: Node, cpus_hint: float = 1, memory_hint: float = 1
    ) -> None:
        self.nodes_by_id = {node.node_id: node for node in nodes}
        self.workers: dict[WorkerId, str] = {}  # worker -> node ID
        self.waiting_for_room: set[WorkerId] = set()
        self.held_by_others: dict[
            str, tuple[float, float]
        ] = {}  # node -> (cpus, memory)
        self.listening: set[WorkerId] = set()
        self.exited: set[WorkerId] = set()  # past their restart limit
        self.hint = (cpus_hint, memory_hint)

    def nodes(self) -> list[Node]:
        nodes = []
        for node_ in self.nodes_by_id.values():
            free_cpus, free_memory = self._free(node_.node_id)
            nodes.append(
                dataclasses.replace(node_, free_cpus=free_cpus, free_memory=free_memory)
            )
        return nodes

    def running(self) -> set[WorkerId]:
        return set(self.workers)

    def waiting(self) -> set[WorkerId]:
        return self.waiting_for_room & set(self.workers)

    def start(self, slot: WorkerSlot) -> None:
        assert self.nodes_by_id[slot.node_id].alive
        if slot.worker_id in self.workers:
            return
        has_room = self._has_room(slot.node_id)
        self.workers[slot.worker_id] = slot.node_id
        if not has_room:
            self.waiting_for_room.add(slot.worker_id)

    def terminate(self, workers: typing.Collection[WorkerId]) -> None:
        for worker in workers:
            self.workers.pop(worker, None)
            self.waiting_for_room.discard(worker)

    def ready(self, workers: typing.Collection[WorkerId]) -> set[WorkerId]:
        if exited := sorted(self.exited.intersection(workers)):
            msg = f"{exited[0]} exited while starting"
            raise WorkerExited(msg)
        return self.listening.intersection(workers)

    def kill(self, worker: WorkerId) -> None:
        del self.workers[worker]

    def remove_node(self, node_id: str) -> None:
        del self.nodes_by_id[node_id]
        self.workers = {n: i for n, i in self.workers.items() if i != node_id}

    def hold(self, node_id: str, cpus: float, memory: float = 0) -> None:
        """Have other Ray work hold `cpus` and `memory` of the node, from now on."""
        self.held_by_others[node_id] = (cpus, memory)
        for worker in sorted(self.waiting()):
            if self.workers[worker] == node_id and self._has_room(node_id):
                self.waiting_for_room.discard(worker)

    def _free(self, node_id: str) -> tuple[float, float]:
        node = self.nodes_by_id[node_id]
        others_cpus, others_memory = self.held_by_others.get(node_id, (0, 0))
        placed = sum(
            1
            for worker, on in self.workers.items()
            if on == node_id and worker not in self.waiting_for_room
        )
        return (
            node.cpus - others_cpus - placed * self.hint[0],
            node.memory - others_memory - placed * self.hint[1],
        )

    def _has_room(self, node_id: str) -> bool:
        cpus, memory = self._free(node_id)
        return cpus >= self.hint[0] and memory >= self.hint[1]


def node(
    node_id: str, *, alive: bool = True, cpus: float = 4, memory: float = 8e9
) -> Node:
    return Node(
        node_id,
        alive=alive,
        cpus=cpus,
        memory=memory,
        free_cpus=cpus,
        free_memory=memory,
    )


def config(**kwargs: typing.Any) -> PolarsRayClusterConfig:
    kwargs.setdefault("worker", PolarsWorkerConfig(cpus_hint=1, memory_hint=1))
    kwargs.setdefault("worker_reconcile_interval", 10)
    kwargs.setdefault("scheduler", PolarsSchedulerConfig())
    return PolarsRayClusterConfig(
        license=PolarsEnterpriseLicenseConfig(license_path="/nonexistent.json"),
        **kwargs,
    )


def pool(workers: InMemoryWorkers, target: int = 0, **kwargs: typing.Any) -> WorkerPool:
    return WorkerPool(workers, config(**kwargs), target)


def test_ensure_puts_one_worker_per_node() -> None:
    workers = InMemoryWorkers(node(NODE_A), node(NODE_B), node(NODE_C))

    assert pool(workers).ensure(2) == {slot(NODE_A), slot(NODE_B)}
    assert workers.running() == {slot(NODE_A), slot(NODE_B)}


def test_a_slot_is_the_worker_of_its_node_prefix_and_index() -> None:
    assert WorkerSlot(NODE_A, 0).worker_id == WorkerId("aaaaaaaaaaaa", 0)
    assert WorkerSlot(NODE_B, 3).worker_id == WorkerId("bbbbbbbbbbbb", 3)


def test_ensure_skips_dead_cpuless_and_too_small_nodes() -> None:
    workers = InMemoryWorkers(
        node(NODE_A, cpus=0),  # e.g. a head started with --num-cpus=0
        node(NODE_B, alive=False),
        node(NODE_C, cpus=2, memory=2e9),
        node("d" * 56, cpus=8, memory=1e9),
    )
    worker = PolarsWorkerConfig(cpus_hint=2, memory_hint=2_000_000_000)

    assert pool(workers, worker=worker).ensure(1) == {slot(NODE_C)}


def test_ensure_spreads_before_stacking() -> None:
    workers = InMemoryWorkers(node(NODE_A), node(NODE_B))

    names = pool(workers, workers_per_node=2).ensure(3)

    assert names == {slot(NODE_A, 0), slot(NODE_B, 0), slot(NODE_A, 1)}


def test_ensure_stacks_no_more_than_a_node_fits() -> None:
    workers = InMemoryWorkers(node(NODE_A, cpus=2), node(NODE_B, cpus=16))

    worker = PolarsWorkerConfig(cpus_hint=2, memory_hint=1)

    names = pool(workers, workers_per_node=3, worker=worker).ensure(4)

    assert names == {slot(NODE_A, 0), slot(NODE_B, 0), slot(NODE_B, 1), slot(NODE_B, 2)}


def test_ensure_reuses_a_freed_slot() -> None:
    workers = InMemoryWorkers(node(NODE_A))
    workers.workers[slot(NODE_A, 1)] = NODE_A

    names = pool(workers, workers_per_node=3).ensure(3)

    assert names == {slot(NODE_A, 0), slot(NODE_A, 1), slot(NODE_A, 2)}


def test_ensure_never_adds_to_or_removes_from_a_larger_pool() -> None:
    workers = InMemoryWorkers(node(NODE_A), node(NODE_B), node(NODE_C))
    worker_pool = pool(workers)
    worker_pool.ensure(3)

    assert worker_pool.ensure(1) == {slot(NODE_A), slot(NODE_B), slot(NODE_C)}


def test_ensure_without_room_starts_nothing() -> None:
    workers = InMemoryWorkers(node(NODE_A), node(NODE_B))

    with pytest.raises(
        NoRoomForWorkers, match="Cannot start 3 workers: 0 running, and room for 2"
    ):
        pool(workers).ensure(3)

    assert workers.running() == set()


def test_rescale_up_starts_what_fits_and_warns(
    caplog: pytest.LogCaptureFixture,
) -> None:
    workers = InMemoryWorkers(node(NODE_A), node(NODE_B))
    worker_pool = pool(workers)

    worker_pool.rescale(3)

    assert workers.running() == {slot(NODE_A), slot(NODE_B)}
    assert worker_pool.status().target == 3
    assert "Only room for 2 of the 3 missing workers" in caplog.text


def test_repeated_rescale_up_adds_no_extra_workers() -> None:
    workers = InMemoryWorkers(node(NODE_A), node(NODE_B), node(NODE_C))
    worker_pool = pool(workers)

    for _ in range(3):
        worker_pool.rescale(2)

    assert workers.running() == {slot(NODE_A), slot(NODE_B)}


@pytest.mark.parametrize(("requested", "clamped"), [(0, 1), (1, 1), (2, 2), (3, 2)])
def test_rescale_is_clamped_to_min_and_max_workers(
    requested: int, clamped: int
) -> None:
    workers = InMemoryWorkers(node(NODE_A), node(NODE_B), node(NODE_C))
    worker_pool = pool(workers, min_workers=1, max_workers=2)

    worker_pool.rescale(requested)

    assert worker_pool.status() == (workers.running(), clamped)
    assert len(workers.running()) == clamped


def test_rescale_to_zero_removes_every_worker() -> None:
    workers = InMemoryWorkers(node(NODE_A), node(NODE_B))
    worker_pool = pool(workers)
    worker_pool.ensure(2)

    worker_pool.rescale(0, keep={str(slot(NODE_A))})

    assert workers.running() == set()


def test_rescale_down_removes_stacked_workers_first() -> None:
    workers = InMemoryWorkers(node(NODE_A), node(NODE_B))
    worker_pool = pool(workers, workers_per_node=2)
    worker_pool.ensure(4)

    worker_pool.rescale(2)

    assert workers.running() == {slot(NODE_A, 0), slot(NODE_B, 0)}


def test_rescale_down_removes_workers_of_lost_nodes_first() -> None:
    workers = InMemoryWorkers(node(NODE_A), node(NODE_B), node(NODE_C))
    worker_pool = pool(workers)
    worker_pool.ensure(3)
    workers.nodes_by_id[NODE_A] = node(NODE_A, alive=False)  # its worker not gone yet

    worker_pool.rescale(2)

    assert workers.running() == {slot(NODE_B), slot(NODE_C)}


def test_rescale_down_deletes_the_workers_named() -> None:
    workers = InMemoryWorkers(node(NODE_A), node(NODE_B), node(NODE_C))
    worker_pool = pool(workers)
    worker_pool.ensure(3)

    # an engine instance ID, an actor name, and a name that is no worker
    worker_pool.rescale(
        1, delete={f"{slot(NODE_A)}-0123abcd", str(slot(NODE_C)), "nope"}
    )

    assert workers.running() == {slot(NODE_B)}


def test_rescale_down_keeps_the_workers_named() -> None:
    workers = InMemoryWorkers(node(NODE_A), node(NODE_B), node(NODE_C))
    worker_pool = pool(workers)
    worker_pool.ensure(3)

    worker_pool.rescale(1, keep={f"{slot(NODE_B)}-0123abcd", "nope"})

    assert workers.running() == {slot(NODE_B)}


def test_rescale_down_deletes_then_keeps() -> None:
    workers = InMemoryWorkers(node(NODE_A), node(NODE_B), node(NODE_C))
    worker_pool = pool(workers)
    worker_pool.ensure(3)

    worker_pool.rescale(
        1, delete={str(slot(NODE_A))}, keep={str(slot(NODE_A)), str(slot(NODE_B))}
    )

    assert workers.running() == {slot(NODE_B)}


def test_rescale_down_ignores_names_that_are_no_workers() -> None:
    workers = InMemoryWorkers(node(NODE_A), node(NODE_B), node(NODE_C))
    worker_pool = pool(workers)
    worker_pool.ensure(3)

    worker_pool.rescale(2, keep={"nope"})

    assert workers.running() == {slot(NODE_A), slot(NODE_B)}


def test_refill_starts_missing_workers_up_to_the_target() -> None:
    workers = InMemoryWorkers(node(NODE_A), node(NODE_B))
    worker_pool = pool(workers, target=2)
    worker_pool.ensure(2)
    workers.kill(slot(NODE_B))

    assert worker_pool.refill(now=0) == [WorkerSlot(NODE_B, 0)]
    assert workers.running() == {slot(NODE_A), slot(NODE_B)}


def test_refill_does_nothing_at_the_target() -> None:
    workers = InMemoryWorkers(node(NODE_A), node(NODE_B))
    worker_pool = pool(workers, target=1)
    worker_pool.ensure(1)

    assert worker_pool.refill(now=0) == []


def test_refill_moves_the_worker_of_a_lost_node_to_a_new_node() -> None:
    workers = InMemoryWorkers(node(NODE_A), node(NODE_B))
    worker_pool = pool(workers, target=1)
    worker_pool.ensure(1)

    workers.remove_node(NODE_A)

    assert worker_pool.refill(now=0) == [WorkerSlot(NODE_B, 0)]


def test_refill_backs_off_a_slot_that_keeps_dying() -> None:
    workers = InMemoryWorkers(node(NODE_A))
    worker_pool = pool(workers, target=1)

    refills = []
    for now in range(0, 200, 5):
        workers.workers.clear()  # it died again right away
        if worker_pool.refill(now=now):
            refills.append(now)

    assert refills == [0, 10, 30, 70, 150]


def test_refill_backoff_is_capped() -> None:
    workers = InMemoryWorkers(node(NODE_A))
    worker_pool = pool(workers, target=1)

    refills = []
    for now in range(0, 100_000, 5):
        workers.workers.clear()
        if worker_pool.refill(now=now):
            refills.append(now)

    assert max(b - a for a, b in itertools.pairwise(refills)) == (MAX_REFILL_BACKOFF)


def test_refill_forgets_the_backoff_of_a_slot_that_stays_up_through_it() -> None:
    workers = InMemoryWorkers(node(NODE_A))
    worker_pool = pool(workers, target=1)

    worker_pool.refill(now=0)  # refill 1, backoff 10 s
    workers.workers.clear()
    worker_pool.refill(now=10)  # refill 2, backoff 20 s
    worker_pool.refill(now=1000)  # still up: backoff forgotten
    workers.workers.clear()

    assert worker_pool.refill(now=1001) == [WorkerSlot(NODE_A, 0)]
    workers.workers.clear()
    assert worker_pool.refill(now=1005) == []
    assert worker_pool.refill(now=1011) == [WorkerSlot(NODE_A, 0)]


def test_refill_uses_another_free_slot_while_one_is_backed_off() -> None:
    workers = InMemoryWorkers(node(NODE_A), node(NODE_B))
    worker_pool = pool(workers, target=1)

    assert worker_pool.refill(now=0) == [WorkerSlot(NODE_A, 0)]
    workers.kill(slot(NODE_A))

    assert worker_pool.refill(now=5) == [WorkerSlot(NODE_B, 0)]


def test_refill_forgets_the_backoff_of_a_lost_node() -> None:
    workers = InMemoryWorkers(node(NODE_A))
    worker_pool = pool(workers, target=1)
    worker_pool.refill(now=0)  # backoff 10 s

    workers.remove_node(NODE_A)
    worker_pool.refill(now=1)
    workers.nodes_by_id[NODE_A] = node(NODE_A)  # back, e.g. a node that rejoined

    assert worker_pool.refill(now=2) == [WorkerSlot(NODE_A, 0)]


def test_refill_warns_only_when_the_shortfall_changes(
    caplog: pytest.LogCaptureFixture,
) -> None:
    workers = InMemoryWorkers(node(NODE_A))
    worker_pool = pool(workers)

    worker_pool.rescale(2)
    worker_pool.refill(now=0)
    worker_pool.refill(now=1)
    worker_pool.rescale(3)

    assert [r.getMessage() for r in caplog.records if "Only room" in r.message] == [
        "Only room for 1 of the 2 missing workers; add nodes or raise workers_per_node",
        "Only room for 0 of the 2 missing workers; add nodes or raise workers_per_node",
    ]


class SlowStart(InMemoryWorkers):
    """Workers whose start blocks until released, as a slow Ray call would."""

    def __init__(self, *nodes: Node) -> None:
        super().__init__(*nodes)
        self.starting = threading.Event()
        self.release = threading.Event()

    def start(self, slot: WorkerSlot) -> None:
        self.starting.set()
        self.release.wait(5)
        super().start(slot)


def test_a_downscale_during_a_refill_is_not_undone() -> None:
    workers = SlowStart(node(NODE_A), node(NODE_B))
    workers.workers[slot(NODE_A)] = NODE_A
    worker_pool = pool(workers, target=2)

    refill = threading.Thread(target=worker_pool.refill, args=(0,))
    refill.start()
    workers.starting.wait(5)
    rescale = threading.Thread(target=worker_pool.rescale, args=(1,))
    rescale.start()
    time.sleep(0.1)  # for the rescale to wait on the refill
    workers.release.set()
    refill.join(5)
    rescale.join(5)
    worker_pool.refill(now=1)

    assert len(workers.running()) == 1


def wait_for(condition: typing.Callable[[], bool]) -> bool:
    deadline = time.monotonic() + 5
    while not condition() and time.monotonic() < deadline:
        time.sleep(0.01)
    return condition()


def test_refilling_runs_in_the_background_until_halted() -> None:
    workers = InMemoryWorkers(node(NODE_A), node(NODE_B))
    worker_pool = WorkerPool(workers, config(worker_reconcile_interval=0.01), 2)

    worker_pool.start_refilling()
    assert wait_for(lambda: len(workers.running()) == 2)

    worker_pool.halt()
    workers.kill(slot(NODE_A))
    time.sleep(0.1)

    assert workers.running() == {slot(NODE_B)}


def test_no_refilling_at_a_zero_interval() -> None:
    workers = InMemoryWorkers(node(NODE_A))
    worker_pool = WorkerPool(workers, config(worker_reconcile_interval=0), 1)

    worker_pool.start_refilling()
    time.sleep(0.1)

    assert workers.running() == set()


def test_stop_removes_every_worker_for_good() -> None:
    workers = InMemoryWorkers(node(NODE_A), node(NODE_B))
    worker_pool = pool(workers)
    worker_pool.ensure(2)

    worker_pool.stop()
    worker_pool.refill(now=0)

    assert workers.running() == set()
    assert worker_pool.status().target == 0


class SlowListing(InMemoryWorkers):
    """Workers that are listed only once they are up, unlike Ray's named actors."""

    def start(self, slot: WorkerSlot) -> None:
        pass


def test_ensure_returns_the_workers_it_started_before_they_are_listed() -> None:
    workers = SlowListing(node(NODE_A), node(NODE_B))
    workers.workers[slot(NODE_A)] = NODE_A

    assert pool(workers).ensure(2) == {slot(NODE_A), slot(NODE_B)}


def test_ensure_raises_a_lower_target() -> None:
    workers = InMemoryWorkers(node(NODE_A), node(NODE_B))
    worker_pool = pool(workers, target=0)

    worker_pool.ensure(2)

    assert worker_pool.status().target == 2


def test_ensure_without_room_leaves_the_target() -> None:
    workers = InMemoryWorkers(node(NODE_A))
    worker_pool = pool(workers, target=1)

    with pytest.raises(NoRoomForWorkers):
        worker_pool.ensure(2)

    assert worker_pool.status().target == 1


def test_a_shortfall_that_comes_back_after_a_downscale_is_reported_again(
    caplog: pytest.LogCaptureFixture,
) -> None:
    workers = InMemoryWorkers(node(NODE_A))
    worker_pool = pool(workers)

    worker_pool.rescale(2)
    worker_pool.rescale(1)
    worker_pool.rescale(2)

    assert caplog.text.count("Only room for") == 2


def test_not_ready_names_the_workers_not_listening_yet() -> None:
    workers = InMemoryWorkers(node(NODE_A), node(NODE_B))
    worker_pool = pool(workers)
    worker_pool.ensure(2)

    workers.listening.add(slot(NODE_A))
    assert worker_pool.not_ready() == {slot(NODE_B)}

    workers.listening.add(slot(NODE_B))
    assert worker_pool.not_ready() == set()


def test_not_ready_awaits_the_workers_started_before_they_are_listed() -> None:
    workers = SlowListing(node(NODE_A), node(NODE_B))
    workers.workers[slot(NODE_A)] = NODE_A
    workers.listening.add(slot(NODE_A))
    worker_pool = pool(workers)

    worker_pool.ensure(2)

    assert worker_pool.not_ready() == {slot(NODE_B)}


def test_not_ready_raises_when_a_started_worker_exited() -> None:
    workers = InMemoryWorkers(node(NODE_A))
    worker_pool = pool(workers)
    worker_pool.ensure(1)

    workers.kill(slot(NODE_A))
    workers.exited.add(slot(NODE_A))

    with pytest.raises(WorkerExited, match=str(slot(NODE_A))):
        worker_pool.not_ready()


def test_not_ready_leaves_a_worker_seen_ready_to_the_refill() -> None:
    workers = InMemoryWorkers(node(NODE_A))
    worker_pool = pool(workers)
    worker_pool.ensure(1)
    workers.listening.add(slot(NODE_A))
    worker_pool.not_ready()

    workers.kill(slot(NODE_A))
    workers.exited.add(slot(NODE_A))

    assert worker_pool.not_ready() == set()


def test_not_ready_forgets_the_workers_stopped() -> None:
    workers = SlowListing(node(NODE_A))
    worker_pool = pool(workers)
    worker_pool.ensure(1)

    worker_pool.stop()

    assert worker_pool.not_ready() == set()


def test_a_stopped_pool_ignores_rescales() -> None:
    workers = InMemoryWorkers(node(NODE_A), node(NODE_B))
    worker_pool = pool(workers)
    worker_pool.ensure(1)
    worker_pool.stop()

    worker_pool.rescale(2)

    assert workers.running() == set()
    assert worker_pool.status().target == 0


def test_ensure_skips_a_node_whose_cpus_other_ray_work_holds() -> None:
    workers = InMemoryWorkers(node(NODE_A), node(NODE_B), cpus_hint=2)
    workers.hold(NODE_A, cpus=3)
    worker = PolarsWorkerConfig(cpus_hint=2, memory_hint=1)

    assert pool(workers, worker=worker).ensure(1) == {slot(NODE_B)}
    assert workers.waiting() == set()


def test_ensure_skips_a_node_whose_memory_other_ray_work_holds() -> None:
    workers = InMemoryWorkers(node(NODE_A), node(NODE_B))
    workers.hold(NODE_A, cpus=0, memory=8e9)

    assert pool(workers).ensure(1) == {slot(NODE_B)}


def test_ensure_stacks_no_more_than_the_free_cpus_fit() -> None:
    workers = InMemoryWorkers(node(NODE_A, cpus=8), cpus_hint=2)
    workers.hold(NODE_A, cpus=4)
    worker = PolarsWorkerConfig(cpus_hint=2, memory_hint=1)
    worker_pool = pool(workers, workers_per_node=4, worker=worker)

    with pytest.raises(NoRoomForWorkers, match="room for 2"):
        worker_pool.ensure(3)

    assert worker_pool.ensure(2) == {slot(NODE_A, 0), slot(NODE_A, 1)}


def test_ensure_counts_the_room_its_own_workers_hold_once() -> None:
    workers = InMemoryWorkers(node(NODE_A, cpus=4), cpus_hint=2)
    worker = PolarsWorkerConfig(cpus_hint=2, memory_hint=1)
    worker_pool = pool(workers, workers_per_node=2, worker=worker)
    worker_pool.ensure(1)

    assert worker_pool.ensure(2) == {slot(NODE_A, 0), slot(NODE_A, 1)}
    assert workers.waiting() == set()


def test_rescale_up_starts_only_what_the_free_room_fits(
    caplog: pytest.LogCaptureFixture,
) -> None:
    workers = InMemoryWorkers(node(NODE_A), node(NODE_B))
    workers.hold(NODE_B, cpus=4)
    worker_pool = pool(workers)

    worker_pool.rescale(2)

    assert workers.running() == {slot(NODE_A)}
    assert "Only room for 1 of the 2 missing workers" in caplog.text


def waiting_on_a(**kwargs: typing.Any) -> tuple[InMemoryWorkers, WorkerPool]:
    """Return a pool of one worker that waits for node A, full, with B free."""
    workers = InMemoryWorkers(node(NODE_A), node(NODE_B))
    worker_pool = pool(workers, target=1, **kwargs)
    workers.hold(NODE_A, cpus=4)  # after the pool saw A free, as other work raced it
    workers.start(WorkerSlot(NODE_A, 0))
    assert workers.waiting() == {slot(NODE_A)}
    return workers, worker_pool


def test_refill_leaves_a_worker_waiting_for_room_within_the_startup_timeout() -> None:
    workers, worker_pool = waiting_on_a(startup_timeout=30)

    assert worker_pool.refill(now=0) == []
    assert worker_pool.refill(now=29) == []
    assert workers.running() == {slot(NODE_A)}


def test_refill_moves_a_worker_waiting_for_room_past_the_startup_timeout(
    caplog: pytest.LogCaptureFixture,
) -> None:
    workers, worker_pool = waiting_on_a(startup_timeout=30)
    worker_pool.refill(now=0)

    assert worker_pool.refill(now=30) == [WorkerSlot(NODE_B, 0)]
    assert workers.running() == {slot(NODE_B)}
    assert (
        f"Removed {slot(NODE_A)}, as it waited 30s for room on its node" in caplog.text
    )


def test_refill_times_the_wait_from_when_it_first_sees_it() -> None:
    workers, worker_pool = waiting_on_a(startup_timeout=30)

    worker_pool.refill(now=100)

    assert workers.running() == {slot(NODE_A)}


def test_refill_restarts_the_wait_of_a_worker_that_got_room() -> None:
    workers, worker_pool = waiting_on_a(startup_timeout=30)
    worker_pool.refill(now=0)
    workers.hold(NODE_A, cpus=0)
    worker_pool.refill(now=10)

    workers.hold(NODE_A, cpus=4)
    assert worker_pool.refill(now=30) == []
    assert workers.running() == {slot(NODE_A)}


def test_waiting_names_the_workers_waiting_for_room() -> None:
    _, worker_pool = waiting_on_a()

    assert worker_pool.waiting() == {slot(NODE_A)}


def test_not_ready_forgets_a_started_worker_removed_for_waiting_for_room() -> None:
    workers = SlowListing(node(NODE_A), node(NODE_B))
    worker_pool = pool(workers, startup_timeout=30)
    worker_pool.ensure(1)
    workers.workers[slot(NODE_A)] = NODE_A  # listed, but Ray holds it pending
    workers.waiting_for_room.add(slot(NODE_A))
    workers.hold(NODE_A, cpus=4)

    worker_pool.refill(now=0)
    worker_pool.refill(now=30)

    assert worker_pool.not_ready() == set()
