import logging
import typing

import ray

from polars_nanook_ray.actors.pool import WorkerPool
from polars_nanook_ray.actors.scaling import serve_scaling
from polars_nanook_ray.actors.utils import (
    _handle_sigterm,
    _log_to_stderr,
)
from polars_nanook_ray.actors.worker import RayWorkers
from polars_nanook_ray.config import PolarsRayClusterConfig
from polars_nanook_ray.identity import SCHEDULER, WorkerId
from polars_nanook_ray.ports import resolve_ports
from polars_nanook_ray.subprocess_runner import SubprocessRunner

if typing.TYPE_CHECKING:
    from polars_nanook_ray.actors.scaling import ScalingEndpoint

logger = logging.getLogger(__name__)


class Endpoints(typing.NamedTuple):
    """The scheduler's URLs: for queries, and for the observatory dashboard, if on."""

    client: str
    dashboard: str | None


@ray.remote
class PolarsSchedulerActor:
    """The central service, distributing tasks and piloting autoscaling."""

    def __init__(self, config: PolarsRayClusterConfig) -> None:
        _log_to_stderr()
        self.config_cluster: PolarsRayClusterConfig = config

        self._runner = SubprocessRunner(config.binary_path, SCHEDULER)

        self.scheduler_host: str = self._runner.host
        self._pool = WorkerPool(
            RayWorkers(config, self.scheduler_host), config, config.num_workers
        )
        self._scaling: ScalingEndpoint | None = None

        self.start()
        _handle_sigterm(self._stop_scheduler)
        self._pool.start_reconciling()

    def __ray_shutdown__(self) -> None:
        self._stop_scheduler()

    def start(self) -> None:
        """Spawn the scheduler process, if not already running.

        Serves scaling requests before launching the engine, which asks for its scaling
        config as it starts.
        """
        if self._runner.is_started():
            return

        resolve_ports(self.config_cluster.config_bind_ports_scheduler())

        toml = self.config_cluster.config_scheduler(self.scheduler_host)

        logger.info(
            "Starting scheduler (%s observatory) on %s",
            "and" if self.config_cluster.scheduler.observatory.enabled else "without",
            self.scheduler_host,
        )

        if self.config_cluster.scheduler.scaling.enabled:
            self._scaling = serve_scaling(self._pool, self.config_cluster)
        try:
            self._runner.start(toml, self.config_cluster.engine_env())
        except BaseException:
            self._stop_scaling()
            raise

    def stop(self) -> None:
        """Stop scaling, the workers, then the scheduler process."""
        self._stop_scaling()
        self._pool.stop()
        self._stop_scheduler()

    def _stop_scheduler(self) -> None:
        """Terminate the scheduler process and clean up; Ray then stops its workers."""
        self._pool.halt()
        self._stop_scaling()
        self._runner.stop()

    def _stop_scaling(self) -> None:
        """Stop serving scaling requests; those in progress may finish."""
        if self._scaling is not None:
            self._scaling.close()

    def is_ready(self) -> bool:
        """Return whether the scheduler process is alive and accepting registration."""
        return self._runner.is_ready(
            self.config_cluster.scheduler.worker_registration_port
        )

    def not_ready(self) -> list[str]:
        """Return the scheduler if it is not ready, else the workers not ready yet.

        Raises `WorkerExited` if a worker exited past its restart limit while starting.
        """
        if not self.is_ready():
            return [str(SCHEDULER)]
        return [str(worker) for worker in sorted(self._pool.not_ready())]

    def get_host(self) -> str:
        """Return the IP address the scheduler is running on."""
        return self.scheduler_host

    def endpoints(self) -> Endpoints:
        """Return the URLs to submit queries to and of the dashboard, if any."""
        scheduler = self.config_cluster.scheduler
        return Endpoints(
            client=f"http://{self.scheduler_host}:{scheduler.client_port}",
            dashboard=(
                f"http://{self.scheduler_host}:{scheduler.observatory.rest_port}"
                if scheduler.observatory.enabled
                else None
            ),
        )

    def get_scheduler_pid(self) -> int | None:
        """Return the OS process ID of the scheduler binary subprocess."""
        return self._runner.pid()

    def ensure_workers(self, num_workers: int) -> set[WorkerId]:
        """Start workers until there are at least `num_workers`; return them all.

        Raises `NoRoomForWorkers`, starting none, if a worker fits no alive node.
        """
        return self._pool.ensure(num_workers)

    def rescale_worker_pool_to(
        self,
        num_workers: int,
        delete: set[str] | None = None,
        keep: set[str] | None = None,
    ) -> None:
        """Rescale the number of workers according to the scheduler requests.

        Parameters
        ----------
        num_workers
            The number of worker to upscale or downscale to, clamped to `min_workers`
            and `max_workers`. If neither `delete` nor `keep` is given, the workers
            with the highest indices are removed.
        delete
            Set of workers to terminate, as engine instance IDs or actor names.
            Applied before `keep`. Values that are not live workers are ignored.
        keep
            Set of workers to keep running, as engine instance IDs or actor names,
            while terminating all the others not already removed via `delete`.
            Values that are not live workers are ignored.

        """
        self._pool.rescale(num_workers, delete=delete, keep=keep)
