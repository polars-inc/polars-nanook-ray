from polars_nanook_ray.actors.placement import WorkerPoolReconciler
from polars_nanook_ray.identity import WorkerId


class Pool:
    def __init__(self, *running: int) -> None:
        self.running = {WorkerId(i) for i in running}
        self.started: list[WorkerId] = []

    def start(self, worker: WorkerId) -> None:
        self.started.append(worker)
        self.running.add(worker)

    def reconciler(self) -> WorkerPoolReconciler:
        return WorkerPoolReconciler(10, self.start, lambda: set(self.running))


def test_starts_missing_workers_up_to_the_target() -> None:
    pool = Pool(0)

    assert pool.reconciler().reconcile(3) == [WorkerId(1), WorkerId(2)]
    assert pool.running == {WorkerId(0), WorkerId(1), WorkerId(2)}


def test_replaces_a_lost_worker_under_its_name() -> None:
    pool = Pool(0, 2)

    assert pool.reconciler().reconcile(3) == [WorkerId(1)]


def test_does_nothing_at_the_target() -> None:
    pool = Pool(0, 1)

    assert pool.reconciler().reconcile(2) == []
    assert pool.reconciler().reconcile(1) == []
