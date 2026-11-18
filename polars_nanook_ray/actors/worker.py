import logging
import typing

import ray
from ray._private.state import actors, available_resources_per_node
from ray.exceptions import ActorDiedError, ActorUnavailableError
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from polars_nanook_ray.actors.pool import Node, WorkerExited, WorkerSlot
from polars_nanook_ray.actors.utils import (
    _handle_sigterm,
    _log_to_stderr,
    resolve_actor_handles,
    terminate_actors,
)
from polars_nanook_ray.config import PolarsRayClusterConfig
from polars_nanook_ray.identity import WorkerId, list_workers
from polars_nanook_ray.ports import resolve_ports
from polars_nanook_ray.subprocess_runner import SubprocessRunner

if typing.TYPE_CHECKING:
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
        slot: WorkerSlot,
        scheduler_host: str,
    ) -> None:
        _log_to_stderr()
        self.config_cluster: PolarsRayClusterConfig = config

        self.worker_id: WorkerId = slot.worker_id
        self.scheduler_host = scheduler_host

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

        logger.info(
            "Starting %s on %s as %s", self.worker_id, self.worker_host, instance_id
        )

        self._runner.start(
            toml,
            {
                **self.config_cluster.engine_env(),
                "POLARS_TEMP_DIR": self.config_cluster.worker.temporary_data_dir,
            },
        )

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
    """The worker actors of one cluster, on the nodes of the Ray cluster."""

    def __init__(self, config: PolarsRayClusterConfig, scheduler_host: str) -> None:
        self._config = config
        self._scheduler_host = scheduler_host
        # Ray destroys an owned actor once its owner holds no handle to it
        self._started: dict[WorkerId, ActorHandle] = {}

    def nodes(self) -> list[Node]:
        free = available_resources_per_node()
        return [
            Node(
                node["NodeID"],
                alive=node["Alive"],
                cpus=node["Resources"].get("CPU", 0),
                memory=node["Resources"].get("memory", 0),
                free_cpus=free.get(node["NodeID"], {}).get("CPU", 0),
                free_memory=free.get(node["NodeID"], {}).get("memory", 0),
            )
            for node in ray.nodes()
        ]

    def running(self) -> set[WorkerId]:
        return list_workers(self._config.cluster_id)

    def waiting(self) -> set[WorkerId]:
        return set(_waiting_for_room(self._handles(self.running())))

    def start(self, slot: WorkerSlot) -> None:
        """Start the worker of `slot` pinned to its node, unless it already runs."""
        self._started[slot.worker_id] = PolarsWorkerActor.options(  # type: ignore[attr-defined]
            name=slot.worker_id.actor_name,
            namespace=self._config.cluster_id,
            get_if_exists=True,
            # owned by the scheduler actor: Ray stops the workers when it dies, and its
            # restart starts new ones that register at its current address
            max_restarts=self._config.worker_max_restarts,
            num_cpus=self._config.worker.cpus_hint,
            memory=self._config.worker.memory_hint,
            scheduling_strategy=NodeAffinitySchedulingStrategy(
                slot.node_id, soft=False
            ),
        ).remote(self._config, slot, self._scheduler_host)

        logger.info("Requested worker %s", slot.worker_id)

    def terminate(self, workers: typing.Collection[WorkerId]) -> None:
        handles = self._handles(workers)
        waiting = _waiting_for_room(handles)
        for handle in waiting.values():  # no process to stop, nor answers to `stop`
            ray.kill(handle)
        terminate_actors(
            [h for w, h in handles.items() if w not in waiting],
            self._config.actor_response_timeout,
        )
        for worker in workers:
            self._started.pop(worker, None)

    def ready(self, workers: typing.Collection[WorkerId]) -> set[WorkerId]:
        handles = self._handles(workers)
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

    def _handles(
        self, workers: typing.Collection[WorkerId]
    ) -> dict[WorkerId, "ActorHandle"]:
        handles = {w: h for w in workers if (h := self._started.get(w)) is not None}
        return handles | resolve_actor_handles(
            set(workers) - set(handles), self._config.cluster_id
        )


def _waiting_for_room(
    handles: dict[WorkerId, "ActorHandle"],
) -> dict[WorkerId, "ActorHandle"]:
    """Return those of `handles` whose actor Ray holds pending for room on its node.

    Ray gives a pending actor its process once its node has room for it.
    """
    pending = actors(actor_state_name="PENDING_CREATION")
    return {
        worker: handle
        for worker, handle in handles.items()
        if (info := pending.get(handle._actor_id.hex())) is not None and not info["Pid"]
    }
