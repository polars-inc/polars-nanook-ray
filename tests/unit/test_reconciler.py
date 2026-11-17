from polars_nanook_ray.actors.placement import (
    MAX_REFILL_BACKOFF,
    WorkerPoolReconciler,
    WorkerSlot,
)
from polars_nanook_ray.identity import WorkerId

SLOT_A = WorkerSlot("a" * 56, 0)
SLOT_B = WorkerSlot("b" * 56, 0)


class Pool:
    def __init__(self, free: list[WorkerSlot]) -> None:
        self.free = free
        self.running: set[WorkerId] = set()
        self.started: list[WorkerId] = []

    def plan(self, target: int, running: set[WorkerId]) -> list[WorkerSlot]:
        missing = target - len(running)
        return [s for s in self.free if s.worker_id not in running][: max(missing, 0)]

    def start(self, slot: WorkerSlot) -> None:
        self.started.append(slot.worker_id)
        self.running.add(slot.worker_id)

    def reconciler(self, interval: float = 10) -> WorkerPoolReconciler:
        return WorkerPoolReconciler(
            interval, self.plan, self.start, lambda: set(self.running)
        )


def test_refills_missing_workers_up_to_the_target() -> None:
    pool = Pool([SLOT_A, SLOT_B])
    pool.running = {SLOT_A.worker_id}

    assert pool.reconciler().reconcile(2, now=0) == [SLOT_B]
    assert pool.running == {SLOT_A.worker_id, SLOT_B.worker_id}


def test_does_nothing_at_the_target() -> None:
    pool = Pool([SLOT_A])
    pool.running = {SLOT_A.worker_id}

    assert pool.reconciler().reconcile(1, now=0) == []


def test_backs_off_a_slot_that_keeps_dying() -> None:
    pool = Pool([SLOT_A])
    reconciler = pool.reconciler(interval=10)

    refills = []
    for now in range(0, 200, 5):
        pool.running.clear()  # it died again right away
        if reconciler.reconcile(1, now=now):
            refills.append(now)

    assert refills == [0, 10, 30, 70, 150]


def test_backoff_is_capped() -> None:
    pool = Pool([SLOT_A])
    reconciler = pool.reconciler(interval=10)

    last = None
    for now in range(0, 100_000, 5):
        pool.running.clear()
        if reconciler.reconcile(1, now=now):
            if last is not None:
                assert now - last <= MAX_REFILL_BACKOFF
            last = now


def test_a_slot_that_stays_up_through_its_backoff_starts_over() -> None:
    pool = Pool([SLOT_A])
    reconciler = pool.reconciler(interval=10)

    reconciler.reconcile(1, now=0)  # refill 1, backoff 10 s
    pool.running.clear()
    reconciler.reconcile(1, now=10)  # refill 2, backoff 20 s
    reconciler.reconcile(1, now=1000)  # still up: backoff forgotten
    pool.running.clear()

    assert reconciler.reconcile(1, now=1001) == [SLOT_A]
    pool.running.clear()
    assert reconciler.reconcile(1, now=1005) == []
    assert reconciler.reconcile(1, now=1011) == [SLOT_A]
