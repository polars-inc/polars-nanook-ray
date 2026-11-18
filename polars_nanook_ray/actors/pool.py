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
    """A Ray node: its resources in all, and those no Ray work holds now."""

    node_id: str
    alive: bool
    cpus: float
    memory: float
    free_cpus: float
    free_memory: float


class Workers(typing.Protocol):
    """Where the worker actors of one cluster live: Ray, or memory in tests."""

    def nodes(self) -> list[Node]: ...

    def running(self) -> set[WorkerId]:
        """Return the live workers."""
        ...

    def waiting(self) -> set[WorkerId]:
        """Return the live workers waiting for their node to have room for them."""
        ...

    def start(self, slot: WorkerSlot) -> None:
        """Start the worker of `slot` on its node, unless it already runs."""
        ...

    def terminate(self, workers: typing.Collection[WorkerId]) -> None:
        """Stop the workers given, ignoring those that are not running."""
        ...

    def ready(self, workers: typing.Collection[WorkerId]) -> set[WorkerId]:
        """Return those of `workers` listening for tasks.

        Raises `WorkerExited` if one of them exited past its restart limit.
        """
        ...


class NoRoomForWorkers(RuntimeError):
    pass


class WorkerExited(RuntimeError):
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
        # started by the last `ensure`, and not seen ready since
        self._starting: set[WorkerId] = set()
        # worker -> when a refill first saw it waiting for room on its node
        self._waiting_since: dict[WorkerId, float] = {}

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
            self._starting = {s.worker_id for s in free[:missing]}
            return running | self._starting

    def not_ready(self) -> set[WorkerId]:
        """Return the workers not listening for tasks yet.

        Checks the running workers, and those the last `ensure` started that are not
        listed yet. Raises `WorkerExited` if one of those exited past its restart
        limit before it was ready.
        """
        with self._lock:
            workers = self._workers.running() | self._starting
        ready = self._workers.ready(workers)
        with self._lock:
            self._starting -= ready
        return workers - ready

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
        either, removes workers on lost nodes first, then the last ones placed. Does
        nothing once halted.
        """
        with self._lock:
            if self._halted.is_set():
                logger.warning("Ignored a rescale to %d workers: halted", num_workers)
                return
            num_workers = self._clamp(num_workers)
            self._target = num_workers  # first, for no refill to undo the removal
            running = self._workers.running()

            if len(running) < num_workers:
                self._start_up_to(num_workers, running, self._free_slots(running))
                return

            self._shortfall = 0
            if len(running) > num_workers:
                remove = self._select_removal(running, num_workers, delete, keep)
                self._workers.terminate(remove)
                self._starting -= remove
                for worker in sorted(remove):
                    logger.info("Removed worker %s", worker)

    def refill(self, now: float) -> list[WorkerSlot]:
        """Start workers on free slots up to the target; return the slots started.

        Ray restarts a worker within its restart limit, but not past it, nor once its
        node is gone. A slot refilled again soon after is backed off exponentially,
        and other free slots are used meanwhile. A worker waiting for room on its node,
        held by other Ray work, for `startup_timeout` is removed, to start on a node
        with room. `now` is in monotonic seconds.
        """
        with self._lock:
            if self._halted.is_set():
                return []

            running = self._workers.running() - self._remove_waiting(now)
            offered = {s.worker_id for s in self._slots()}
            for worker, (_, retry_at) in list(self._backoff.items()):
                if worker not in offered or (worker in running and now >= retry_at):
                    del self._backoff[worker]  # its node is gone, or it stayed up

            free = self._free_slots(running)
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

    def waiting(self) -> set[WorkerId]:
        """Return the workers waiting for their node to have room for them."""
        return self._workers.waiting()

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
        """Stop refilling and rescaling, for good; leave the workers running."""
        self._halted.set()

    def stop(self) -> None:
        """Stop refilling, then every worker."""
        self.halt()
        with self._lock:
            self._target = 0
            self._starting.clear()
            running = self._workers.running()
            if running:
                logger.info("Stopping %d worker(s)...", len(running))
                self._workers.terminate(running)

    def status(self) -> PoolStatus:
        """Return the running workers, and the target.

        Does not wait for a rescale or refill in progress.
        """
        return PoolStatus(self._workers.running(), self._target)

    def _refill_until_halted(self, interval: float) -> None:
        while not self._halted.wait(interval):
            try:
                self.refill(time.monotonic())
            except Exception:
                logger.exception("Could not refill the worker pool")

    def _remove_waiting(self, now: float) -> set[WorkerId]:
        """Remove the workers waiting for room for `startup_timeout`; return them."""
        waiting = self._workers.waiting()
        self._waiting_since = {
            worker: self._waiting_since.get(worker, now) for worker in waiting
        }
        timeout = self._config.startup_timeout
        removed = {
            worker
            for worker, since in self._waiting_since.items()
            if now - since >= timeout
        }
        if removed:
            self._workers.terminate(removed)
            self._starting -= removed
            for worker in sorted(removed):
                del self._waiting_since[worker]
                logger.warning(
                    "Removed %s, as it waited %ds for room on its node, which other "
                    "Ray work holds",
                    worker,
                    timeout,
                )
        return removed

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
            if (capacity := self._capacity(node)) > 0
        }
        return [
            WorkerSlot(node_id, index)
            for index in range(max(capacities.values(), default=0))
            for node_id in sorted(capacities)
            if index < capacities[node_id]
        ]

    def _capacity(self, node: Node) -> int:
        """Return how many workers `node` has room for."""
        if not node.alive or node.cpus <= 0:
            return 0
        return int(
            min(
                self._config.workers_per_node,
                _fits(node.cpus, self._config.worker.cpus_hint),
                _fits(node.memory, self._config.worker.memory_hint),
            )
        )

    def _free_slots(self, running: typing.Collection[WorkerId]) -> list[WorkerSlot]:
        """The slots without a worker that their node's free resources fit."""
        room = {node.node_id: self._room(node) for node in self._workers.nodes()}
        free = []
        for worker_slot in self._slots():
            node_id = worker_slot.node_id
            if worker_slot.worker_id not in running and room.get(node_id, 0) >= 1:
                room[node_id] -= 1
                free.append(worker_slot)
        return free

    def _room(self, node: Node) -> float:
        """Return how many more workers the free resources of `node` fit."""
        return min(
            _fits(node.free_cpus, self._config.worker.cpus_hint),
            _fits(node.free_memory, self._config.worker.memory_hint),
        )


def _live_workers_of(
    requested: typing.Collection[str] | None, running: set[WorkerId]
) -> set[WorkerId] | None:
    """Return the running workers the engine instance IDs or actor names given are."""
    return running & {identify(value) for value in requested or ()} or None
