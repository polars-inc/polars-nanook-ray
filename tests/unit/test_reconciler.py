from polars_nanook_ray.actors.placement import WorkerPoolReconciler


class Pool:
    def __init__(self, *running: str) -> None:
        self.running = set(running)
        self.started: list[str] = []

    def start(self, name: str) -> None:
        self.started.append(name)
        self.running.add(name)

    def reconciler(self) -> WorkerPoolReconciler:
        return WorkerPoolReconciler(10, self.start, lambda: set(self.running))


def test_starts_missing_workers_up_to_the_target() -> None:
    pool = Pool("worker-0")

    assert pool.reconciler().reconcile(3) == ["worker-1", "worker-2"]
    assert pool.running == {"worker-0", "worker-1", "worker-2"}


def test_replaces_a_lost_worker_under_its_name() -> None:
    pool = Pool("worker-0", "worker-2")

    assert pool.reconciler().reconcile(3) == ["worker-1"]


def test_does_nothing_at_the_target() -> None:
    pool = Pool("worker-0", "worker-1")

    assert pool.reconciler().reconcile(2) == []
    assert pool.reconciler().reconcile(1) == []
