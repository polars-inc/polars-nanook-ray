import dataclasses
import logging
import threading
import time
import typing

from polars_nanook_ray.identity import WorkerId, identify

if typing.TYPE_CHECKING:
    from polars_nanook_ray.config import PolarsRayClusterConfig

MAX_REFILL_BACKOFF = 600.0

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class WorkerSlot:
    """A place for one worker: a Ray node, and its position on that node."""

    node_id: str
    index: int

    @property
    def worker_id(self) -> WorkerId:
        return WorkerId.of(self.node_id, self.index)


@dataclasses.dataclass(frozen=True)
class Node:
    node_id: str
    alive: bool
    cpus: float
    memory: float


class Workers(typing.Protocol):
    """Where the worker actors of one cluster live: Ray, or memory in tests."""

    def nodes(self) -> list[Node]: ...

    def running(self) -> set[WorkerId]:
        """Return the live workers."""
        ...

    def start(self, slot: WorkerSlot) -> None:
        """Start the worker of `slot` on its node, unless it already runs."""
        ...

    def terminate(self, workers: typing.Collection[WorkerId]) -> None:
        """Stop the workers given, ignoring those that are not running."""
        ...


class NoRoomForWorkers(RuntimeError):
    pass


class PoolStatus(typing.NamedTuple):
    running: set[WorkerId]
    target: int


def _fits(total: float, request: float) -> float:
    return total // request if request > 0 else float("inf")


class WorkerPool:
    """The workers of one cluster: how many there should be, and where they run."""

    def __init__(
        self,
        workers: Workers,
        config: "PolarsRayClusterConfig",
        target: int,
    ) -> None:
        self._workers = workers
        self._config = config
        self._target = target
        self._lock = threading.Lock()
        self._shortfall = 0
        # slot -> (refills in a row, earliest time to refill it again)
        self._backoff: dict[WorkerId, tuple[int, float]] = {}
        self._halted = threading.Event()

    def ensure(self, num_workers: int) -> set[WorkerId]:
        """Start workers until there are at least `num_workers`; return them all."""
        with self._lock:
            running = self._workers.running()
            missing = max(num_workers - len(running), 0)
            free = self._free_slots(running)
            if len(free) < missing:
                msg = (
                    f"Cannot start {num_workers} workers: {len(running)} running, and "
                    f"room for {len(free)} more; add nodes or raise workers_per_node."
                )
                raise NoRoomForWorkers(msg)
            for worker_slot in free[:missing]:
                self._workers.start(worker_slot)
            self._target = max(self._target, num_workers)
            return self._workers.running()

    def rescale(
        self,
        num_workers: int,
        delete: typing.Collection[str] | None = None,
        keep: typing.Collection[str] | None = None,
    ) -> None:
        """Start or remove workers for the pool to reach `num_workers`.

        `num_workers` is clamped to `min_workers` and `max_workers`. Starts as many
        missing workers as the nodes have room for. When removing, `delete` names
        workers to remove, then `keep` workers to keep while removing all the others;
        both take engine instance IDs or actor names, and ignore other names. Without
        either, removes workers on lost nodes first, then the last ones placed.
        """
        with self._lock:
            num_workers = self._clamp(num_workers)
            self._target = num_workers  # first, for no refill to undo the removal
            running = self._workers.running()

            if len(running) < num_workers:
                self._start_up_to(num_workers, running, self._free_slots(running))
            elif len(running) > num_workers:
                remove = self._select_removal(running, num_workers, delete, keep)
                self._workers.terminate(remove)
                for worker in sorted(remove):
                    logger.info("Removed worker %s", worker)

    def refill(self, now: float) -> list[WorkerSlot]:
        """Start workers on free slots up to the target; return the slots started.

        Ray restarts a worker within its restart limit, but not past it, nor once its
        node is gone. A slot refilled again soon after is backed off exponentially,
        and other free slots are used meanwhile. `now` is in monotonic seconds.
        """
        with self._lock:
            if self._halted.is_set():
                return []

            running = self._workers.running()
            slots = self._slots()
            offered = {s.worker_id for s in slots}
            for worker, (_, retry_at) in list(self._backoff.items()):
                if worker not in offered or (worker in running and now >= retry_at):
                    del self._backoff[worker]  # its node is gone, or it stayed up

            free = [s for s in slots if s.worker_id not in running]
            ready = [
                s for s in free if now >= self._backoff.get(s.worker_id, (0, 0.0))[1]
            ]
            started = self._start_up_to(self._target, running, free, ready)

            interval = self._config.worker_reconcile_interval
            for worker_slot in started:
                logger.warning("Refilled worker slot %s", worker_slot.worker_id)
                refills, _ = self._backoff.get(worker_slot.worker_id, (0, 0.0))
                delay = min(interval * 2**refills, MAX_REFILL_BACKOFF)
                self._backoff[worker_slot.worker_id] = (refills + 1, now + delay)
            return started

    def start_refilling(self) -> None:
        """Refill every `worker_reconcile_interval` seconds in a thread until halted."""
        interval = self._config.worker_reconcile_interval
        if interval > 0:
            threading.Thread(
                target=self._refill_until_halted,
                args=(interval,),
                name="refill-workers",
                daemon=True,
            ).start()

    def halt(self) -> None:
        """Stop refilling, for good; leave the workers running."""
        self._halted.set()

    def stop(self) -> None:
        """Stop refilling, then every worker."""
        self.halt()
        with self._lock:
            self._target = 0
            running = self._workers.running()
            if running:
                logger.info("Stopping %d worker(s)...", len(running))
                self._workers.terminate(running)

    def status(self) -> PoolStatus:
        """Return the running workers, and the target."""
        with self._lock:
            return PoolStatus(self._workers.running(), self._target)

    def _refill_until_halted(self, interval: float) -> None:
        while not self._halted.wait(interval):
            try:
                self.refill(time.monotonic())
            except Exception:
                logger.exception("Could not refill the worker pool")

    def _clamp(self, num_workers: int) -> int:
        low, high = self._config.min_workers, self._config.max_workers
        clamped = max(low, num_workers if high is None else min(high, num_workers))
        if clamped != num_workers:
            logger.warning(
                "Clamped requested %d workers to %d, within [%d, %s]",
                num_workers,
                clamped,
                low,
                high,
            )
        return clamped

    def _start_up_to(
        self,
        num_workers: int,
        running: set[WorkerId],
        free: list[WorkerSlot],
        ready: list[WorkerSlot] | None = None,
    ) -> list[WorkerSlot]:
        """Start the first `ready` slots for the pool to reach `num_workers`.

        Warns when the shortfall, the missing workers the `free` slots leave out,
        changes.
        """
        missing = max(num_workers - len(running), 0)
        shortfall = max(missing - len(free), 0)
        if shortfall != self._shortfall and shortfall > 0:
            logger.warning(
                "Only room for %d of the %d missing workers; add nodes or raise "
                "workers_per_node",
                len(free),
                missing,
            )
        self._shortfall = shortfall

        started = (free if ready is None else ready)[:missing]
        for worker_slot in started:
            self._workers.start(worker_slot)
        return started

    def _select_removal(
        self,
        running: set[WorkerId],
        num_workers: int,
        delete: typing.Collection[str] | None,
        keep: typing.Collection[str] | None,
    ) -> set[WorkerId]:
        if num_workers == 0:
            return running

        to_delete = _live_workers_of(delete, running)
        to_keep = _live_workers_of(keep, running)
        if to_delete is None and to_keep is None:
            placed = [s.worker_id for s in self._slots() if s.worker_id in running]
            lost = sorted(running - set(placed))
            return set((lost + placed[::-1])[: len(running) - num_workers])

        remove = to_delete or set()
        if to_keep is not None:
            remove |= running - remove - to_keep
        return remove

    def _slots(self) -> list[WorkerSlot]:
        """Every slot the nodes have room for, spread across nodes first."""
        capacities = {
            node.node_id: capacity
            for node in self._workers.nodes()
            if node.alive
            and node.cpus > 0
            and (
                capacity := int(
                    min(
                        self._config.workers_per_node,
                        _fits(node.cpus, self._config.worker.cpus_hint),
                        _fits(node.memory, self._config.worker.memory_hint),
                    )
                )
            )
            > 0
        }
        return [
            WorkerSlot(node_id, index)
            for index in range(max(capacities.values(), default=0))
            for node_id in sorted(capacities)
            if index < capacities[node_id]
        ]

    def _free_slots(self, running: typing.Collection[WorkerId]) -> list[WorkerSlot]:
        return [s for s in self._slots() if s.worker_id not in running]


def _live_workers_of(
    requested: typing.Collection[str] | None, running: set[WorkerId]
) -> set[WorkerId] | None:
    """Return the running workers the engine instance IDs or actor names given are."""
    return running & {identify(value) for value in requested or ()} or None
