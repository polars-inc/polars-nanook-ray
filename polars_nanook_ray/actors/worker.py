import logging
import os
import typing

import ray
from ray.exceptions import ActorDiedError, ActorUnavailableError

from polars_nanook_ray.actors.pool import Node, WorkerExited
from polars_nanook_ray.actors.utils import (
    _handle_sigterm,
    _remove_config,
    _resolve_config_path,
    _write_config,
    resolve_actor_handles,
    terminate_actors,
)
from polars_nanook_ray.config import PolarsRayClusterConfig
from polars_nanook_ray.identity import WorkerId, list_workers
from polars_nanook_ray.ports import resolve_ports
from polars_nanook_ray.subprocess_runner import SubprocessRunner

if typing.TYPE_CHECKING:
    import pathlib

    from ray.actor import ActorHandle


logger = logging.getLogger(__name__)

# a worker still starting answers once it is constructed
READY_CHECK_TIMEOUT = 2.0


@ray.remote
class PolarsWorkerActor:
    """The computing service, accepting tasks from the scheduler."""

    def __init__(
        self,
        config: PolarsRayClusterConfig,
        worker_id: WorkerId,
        scheduler_host: str,
    ) -> None:
        self.config_cluster: PolarsRayClusterConfig = config

        self.worker_id = worker_id
        self.scheduler_host = scheduler_host

        self._config_path: pathlib.Path = _resolve_config_path(self.worker_id)
        self._runner = SubprocessRunner(config.binary_path, self.worker_id)

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

        self._runner.stop_orphans(self.config_cluster.cluster_id)

        ports = resolve_ports(
            {
                "task_port": self.config_cluster.worker.task_port,
                "shuffle_port": self.config_cluster.worker.shuffle_port,
            }
        )
        self.task_port = ports["task_port"]

        # Unique per start: a restarted worker registers as a new one rather than as the
        # worker it replaces, whose registration expires once its heartbeats stop.
        instance_id = self.worker_id.new_instance_id()
        toml = self.config_cluster.config_worker(
            instance_id,
            self.worker_host,
            self.scheduler_host,
            task_port=ports["task_port"],
            shuffle_port=ports["shuffle_port"],
        )
        _write_config(self._config_path, toml)

        logger.info(
            "Starting %s on %s as %s", self.worker_id, self.worker_host, instance_id
        )

        env: dict[str, str] = {
            **os.environ,
            "POLARS_EULA_ACCEPTED": "yes" if self.config_cluster.accept_eula else "no",
            "POLARS_TEMP_DIR": self.config_cluster.worker.temporary_data_dir,
            "PLC_LOG_LEVEL": self.config_cluster.log_level,
            "RUST_BACKTRACE": self.config_cluster.rust_backtrace,
        }

        self._runner.start(self._config_path, env)

    def stop(self) -> None:
        """Terminate the worker process and clean up."""
        self._runner.stop()

        _remove_config(self._config_path)

    def is_ready(self) -> bool:
        """Return whether the worker process is alive and listening for tasks."""
        self._runner._raise_if_exited()

        return (
            self.task_port is not None
            and self._runner.is_alive()
            and self._runner.is_listening(self.task_port)
        )

    def get_worker_pid(self) -> int | None:
        """Return the OS process ID of the worker binary subprocess."""
        return self._runner.pid()


class RayWorkers:
    """The worker actors of one cluster, spread by Ray over the nodes with room."""

    def __init__(
        self, config: PolarsRayClusterConfig, scheduler_host: str | None = None
    ) -> None:
        self._config = config
        self._scheduler_host = scheduler_host
        # Ray destroys an owned actor once its owner holds no handle to it
        self._started: dict[WorkerId, ActorHandle] = {}

    def nodes(self) -> list[Node]:
        return [
            Node(node["Resources"].get("CPU", 0), node["Resources"].get("memory", 0))
            for node in ray.nodes()
            if node["Alive"]
        ]

    def running(self) -> set[WorkerId]:
        return list_workers(self._config.cluster_id)

    def start(self, worker: WorkerId) -> None:
        """Request `worker` from Ray, unless it already exists.

        Ray places it on a node with its CPUs and memory free, spreading the workers
        over the nodes as best it can; until one has room, it waits, as demand Ray's
        autoscaler can add nodes for.
        """
        if self._scheduler_host is None:
            msg = "Cannot start workers without the scheduler's host"
            raise RuntimeError(msg)

        self._started[worker] = PolarsWorkerActor.options(  # type: ignore[attr-defined]
            name=worker.actor_name,
            namespace=self._config.cluster_id,
            get_if_exists=True,
            # owned by the scheduler actor: Ray stops the workers when it dies, and its
            # restart starts new ones that register at its current address
            max_restarts=self._config.worker_max_restarts,
            num_cpus=self._config.worker.cpus_hint,
            memory=self._config.worker.memory_hint,
            scheduling_strategy="SPREAD",
        ).remote(self._config, worker, self._scheduler_host)

        logger.info("Requested worker %s", worker)

    def terminate(self, workers: typing.Collection[WorkerId]) -> None:
        """Stop the workers given; kill those that cannot answer at once.

        A worker waiting for room on a node never answers.
        """
        actors = list(resolve_actor_handles(workers, self._config.cluster_id).values())
        if not actors:
            return
        checks: dict[ray.ObjectRef[None], ActorHandle] = {
            actor.__ray_ready__.remote(): actor for actor in actors
        }
        answered, _ = ray.wait(
            list(checks), num_returns=len(checks), timeout=READY_CHECK_TIMEOUT
        )
        placed = [checks[check] for check in answered]
        terminate_actors(placed, self._config.actor_response_timeout)
        for actor in actors:
            ray.kill(actor)
        for worker in workers:
            self._started.pop(worker, None)

    def ready(self, workers: typing.Collection[WorkerId]) -> set[WorkerId]:
        handles = {w: h for w in workers if (h := self._started.get(w)) is not None}
        handles |= resolve_actor_handles(
            set(workers) - set(handles), self._config.cluster_id
        )
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
