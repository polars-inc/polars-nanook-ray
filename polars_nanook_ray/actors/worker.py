import logging
import typing

import ray
from ray.exceptions import ActorDiedError, ActorUnavailableError

from polars_nanook_ray._engine import (
    SchedulerAddress,
    render_worker,
    worker_dirs,
    worker_env,
    worker_ports,
)
from polars_nanook_ray.actors.pool import Node, WorkerExited
from polars_nanook_ray.actors.utils import (
    ACTOR_RESPONSE_TIMEOUT,
    _handle_sigterm,
    _log_to_stderr,
    make_dirs,
    resolve_actor_handles,
    terminate_actors,
)
from polars_nanook_ray.identity import WorkerId, list_workers
from polars_nanook_ray.ports import resolve_ports
from polars_nanook_ray.subprocess_runner import SubprocessRunner

if typing.TYPE_CHECKING:
    from ray.actor import ActorHandle

    from polars_nanook_ray._spec import ClusterSpec


logger = logging.getLogger(__name__)

# a worker still starting answers once it is constructed
READY_CHECK_TIMEOUT = 2.0


@ray.remote
class PolarsWorkerActor:
    """The computing service, accepting tasks from the scheduler."""

    def __init__(
        self,
        spec: "ClusterSpec",
        worker_id: WorkerId,
        scheduler: SchedulerAddress,
    ) -> None:
        _log_to_stderr()
        self._spec = spec

        self.worker_id = worker_id
        self._scheduler = scheduler

        self._runner = SubprocessRunner(spec.binary_path, self.worker_id)

        self.worker_host: str = self._runner.host
        self.task_port: int | None = None

        self.start()
        _handle_sigterm(self.stop)

    def __ray_shutdown__(self) -> None:
        self.stop()

    def start(self) -> None:
        """Spawn the worker process, if not already running."""
        if self._runner.is_started():
            return

        ports = resolve_ports(worker_ports(self._spec))
        self.task_port = ports["task"]
        make_dirs(worker_dirs(self._spec))

        # Unique per start: a restarted worker registers as a new one rather than as the
        # worker it replaces, whose registration expires once its heartbeats stop.
        instance_id = self.worker_id.new_instance_id()
        toml = render_worker(
            self._spec,
            instance_id=instance_id,
            host=self.worker_host,
            task_port=ports["task"],
            shuffle_port=ports["shuffle"],
            scheduler=self._scheduler,
        )

        logger.info(
            "Starting %s on %s as %s", self.worker_id, self.worker_host, instance_id
        )

        self._runner.start(toml, worker_env(self._spec))

    def stop(self) -> None:
        """Terminate the worker process and clean up."""
        self._runner.stop()

    def is_ready(self) -> bool:
        """Return whether the worker process is alive and listening for tasks."""
        return self.task_port is not None and self._runner.is_ready(self.task_port)

    def get_worker_pid(self) -> int | None:
        """Return the OS process ID of the worker binary subprocess."""
        return self._runner.pid()


class RayWorkers:
    """The worker actors of one cluster, spread by Ray over the nodes with room."""

    def __init__(self, spec: "ClusterSpec", scheduler: SchedulerAddress) -> None:
        self._spec = spec
        self._scheduler = scheduler
        # Ray destroys an owned actor once its owner holds no handle to it
        self._started: dict[WorkerId, ActorHandle] = {}

    def nodes(self) -> list[Node]:
        return [
            Node(node["Resources"].get("CPU", 0), node["Resources"].get("memory", 0))
            for node in ray.nodes()
            if node["Alive"]
        ]

    def running(self) -> set[WorkerId]:
        return list_workers(self._spec.name)

    def start(self, worker: WorkerId) -> None:
        """Request `worker` from Ray, unless it already exists.

        Ray places it on a node with its CPUs and memory free, spreading the workers
        over the nodes as best it can; until one has room, it waits, as demand Ray's
        autoscaler can add nodes for.
        """
        self._started[worker] = PolarsWorkerActor.options(  # type: ignore[attr-defined]
            name=worker.actor_name,
            namespace=self._spec.name,
            get_if_exists=True,
            # owned by the scheduler actor: Ray stops the workers when it dies, and its
            # restart starts new ones that register at its current address
            max_restarts=self._spec.worker_max_restarts,
            num_cpus=self._spec.cpus,
            memory=self._spec.memory_bytes,
            scheduling_strategy="SPREAD",
        ).remote(self._spec, worker, self._scheduler)

        logger.info("Requested worker %s", worker)

    def terminate(self, workers: typing.Collection[WorkerId]) -> None:
        """Stop the workers given; kill those that cannot answer at once.

        A worker waiting for room on a node never answers, and a worker's engine
        stops when its actor dies.
        """
        actors = list(resolve_actor_handles(workers, self._spec.name).values())
        if not actors:
            return
        checks: dict[ray.ObjectRef[None], ActorHandle] = {
            actor.__ray_ready__.remote(): actor for actor in actors
        }
        answered, _ = ray.wait(
            list(checks), num_returns=len(checks), timeout=READY_CHECK_TIMEOUT
        )
        placed = [checks[check] for check in answered]
        terminate_actors(placed, ACTOR_RESPONSE_TIMEOUT)
        for actor in actors:
            ray.kill(actor)
        for worker in workers:
            self._started.pop(worker, None)

    def ready(self, workers: typing.Collection[WorkerId]) -> set[WorkerId]:
        handles = {w: h for w in workers if (h := self._started.get(w)) is not None}
        handles |= resolve_actor_handles(set(workers) - set(handles), self._spec.name)
        if not handles:
            return set()

        checks: dict[WorkerId, ray.ObjectRef[bool]] = {
            worker: handle.is_ready.remote() for worker, handle in handles.items()
        }
        answered, _ = ray.wait(
            list(checks.values()), num_returns=len(checks), timeout=READY_CHECK_TIMEOUT
        )

        ready: set[WorkerId] = set()
        for worker, check in checks.items():
            if check not in answered:
                continue
            try:
                if ray.get(check):
                    ready.add(worker)
            except ActorUnavailableError:  # Ray is restarting it: not ready yet
                continue
            except ActorDiedError as exc:
                msg = (
                    f"{worker} exited while starting, past its restart limit; see its "
                    f"actor's log for the engine's error\n{exc}"
                )
                raise WorkerExited(msg) from exc
        return ready
