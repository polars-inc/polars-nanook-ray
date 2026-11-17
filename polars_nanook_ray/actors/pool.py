import itertools
import logging
import threading
import typing

from polars_nanook_ray.identity import WorkerId, identify

if typing.TYPE_CHECKING:
    from polars_nanook_ray.config import PolarsRayClusterConfig

logger = logging.getLogger(__name__)


class Node(typing.NamedTuple):
    """The total CPUs and memory, in bytes, of an alive Ray node."""

    cpus: float
    memory: float


class Workers(typing.Protocol):
    """Where the worker actors of one cluster live: Ray, or memory in tests."""

    def nodes(self) -> list[Node]:
        """Return the alive nodes."""
        ...

    def running(self) -> set[WorkerId]:
        """Return the workers that exist, also those waiting for room on a node."""
        ...

    def start(self, worker: WorkerId) -> None:
        """Start `worker` on a node Ray picks, unless it already exists."""
        ...

    def terminate(self, workers: typing.Collection[WorkerId]) -> None:
        """Stop the workers given, ignoring those that are not running."""
        ...


class NoRoomForWorkers(RuntimeError):
    pass


class PoolStatus(typing.NamedTuple):
    running: set[WorkerId]
    target: int


class WorkerPool:
    """The workers of one cluster, kept at a target count; Ray places them."""

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
        self._halted = threading.Event()

    def ensure(self, num_workers: int) -> set[WorkerId]:
        """Start workers until there are at least `num_workers`; return them all.

        Raises `NoRoomForWorkers`, starting none, if a worker's CPUs and memory exceed
        those of every alive node: Ray would never place it. Workers that fit a node
        that is busy wait until Ray has room for them.
        """
        with self._lock:
            running = self._workers.running()
            if num_workers > len(running):
                self._check_room()
            self._start_up_to(num_workers, running)
            self._target = max(self._target, num_workers)
            return self._workers.running()

    def rescale(
        self,
        num_workers: int,
        delete: typing.Collection[str] | None = None,
        keep: typing.Collection[str] | None = None,
    ) -> None:
        """Start or remove workers for the pool to reach `num_workers`.

        `num_workers` is clamped to `min_workers` and `max_workers`. When removing,
        `delete` names workers to remove, then `keep` workers to keep while removing
        all the others; both take engine instance IDs or actor names, and ignore other
        names. Without either, removes the highest indices first.
        """
        with self._lock:
            num_workers = self._clamp(num_workers)
            self._target = num_workers  # first, for no reconcile to undo the removal
            running = self._workers.running()
            if len(running) < num_workers:
                self._start_up_to(num_workers, running)
            elif len(running) > num_workers:
                remove = _select_removal(running, num_workers, delete, keep)
                self._workers.terminate(remove)
                for worker in sorted(remove):
                    logger.info("Removed worker %s", worker)

    def reconcile(self) -> list[WorkerId]:
        """Start workers up to the target; return those started.

        Replaces workers Ray no longer restarts: past their restart limit. Does
        nothing once halted.
        """
        with self._lock:
            if self._halted.is_set():
                return []
            started = self._start_up_to(self._target, self._workers.running())
            for worker in started:
                logger.warning("Replaced worker %s", worker)
            return started

    def start_reconciling(self) -> None:
        """Reconcile in a thread until halted.

        Every `worker_reconcile_interval` seconds; never if that is 0.
        """
        interval = self._config.worker_reconcile_interval
        if interval > 0:
            threading.Thread(
                target=self._reconcile_until_halted,
                args=(interval,),
                name="reconcile-workers",
                daemon=True,
            ).start()

    def halt(self) -> None:
        """Stop reconciling, for good; leave the workers running."""
        self._halted.set()

    def stop(self) -> None:
        """Stop reconciling, then every worker."""
        self.halt()
        with self._lock:
            self._target = 0
            running = self._workers.running()
            if running:
                logger.info("Stopping %d worker(s)...", len(running))
                self._workers.terminate(running)

    def status(self) -> PoolStatus:
        """Return the existing workers, and the target."""
        with self._lock:
            return PoolStatus(self._workers.running(), self._target)

    def _reconcile_until_halted(self, interval: float) -> None:
        while not self._halted.wait(interval):
            try:
                self.reconcile()
            except Exception:
                logger.exception("Could not reconcile the worker pool")

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

    def _check_room(self) -> None:
        cpus, memory = self._config.worker.cpus_hint, self._config.worker.memory_hint
        nodes = self._workers.nodes()
        if any(n.cpus >= cpus and n.memory >= memory for n in nodes):
            return
        most = Node(
            max((n.cpus for n in nodes), default=0),
            max((n.memory for n in nodes), default=0),
        )
        if cpus > most.cpus:
            shortfall = f"{cpus:g} CPU, but the largest node has {most.cpus:g} CPU"
        elif memory > most.memory:
            shortfall = f"{_gib(memory)} of memory, but the largest node has {_gib(most.memory)}"
        else:
            shortfall = f"{cpus:g} CPU and {_gib(memory)} of memory, but no single node has both"
        msg = (
            f"Cannot place a worker on any alive Ray node: it requests {shortfall}. "
            "Lower `cpus_hint` or `memory_hint`, or add larger nodes."
        )
        raise NoRoomForWorkers(msg)

    def _start_up_to(self, num_workers: int, running: set[WorkerId]) -> list[WorkerId]:
        """Start workers at the lowest free indices for `num_workers` to exist."""
        missing = max(num_workers - len(running), 0)
        free = (w for i in itertools.count() if (w := WorkerId(i)) not in running)
        started = list(itertools.islice(free, missing))
        for worker in started:
            self._workers.start(worker)
        return started


def _select_removal(
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
        return set(sorted(running)[num_workers:])
    remove = to_delete or set()
    if to_keep is not None:
        remove |= running - remove - to_keep
    return remove


def _live_workers_of(
    requested: typing.Collection[str] | None, running: set[WorkerId]
) -> set[WorkerId] | None:
    """Return the running workers the engine instance IDs or actor names given are."""
    return running & {identify(value) for value in requested or ()} or None


def _gib(memory: float) -> str:
    return f"{memory / 1024**3:g} GiB"
