import logging
import os
import re
import typing

import ray

from polars_nanook_ray.actors.ports import resolve_ports
from polars_nanook_ray.actors.subprocess_runner import SubprocessRunner, service_args
from polars_nanook_ray.actors.utils import (
    _handle_sigterm,
    _remove_config,
    _resolve_config_path,
    _write_config,
)
from polars_nanook_ray.config import PolarsRayClusterConfig

if typing.TYPE_CHECKING:
    import pathlib


WORKER_NAME_PREFIX = "worker"

logger = logging.getLogger(__name__)


def resolve_worker_name(worker_id: int) -> str:
    return f"{WORKER_NAME_PREFIX}-{worker_id}"


def _resolve_worker_name_regex() -> re.Pattern:
    return re.compile(rf"^{WORKER_NAME_PREFIX}-(\d+)$")


@ray.remote
class PolarsWorkerActor:
    """The computing service, accepting tasks from the scheduler."""

    def __init__(
        self,
        config: PolarsRayClusterConfig,
        worker_id: int,
        scheduler_host: str,
    ) -> None:
        self.config_cluster: PolarsRayClusterConfig = config

        self.worker_id = worker_id
        self.worker_name: str = resolve_worker_name(worker_id)
        self.scheduler_host = scheduler_host

        self._config_path: pathlib.Path = _resolve_config_path(self.worker_name)
        self._runner = SubprocessRunner(
            config.binary_path,
            self.worker_name,
            config.instance_name(self.worker_name),
        )

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

        toml = self.config_cluster.config_worker(
            self.worker_id,
            self.worker_host,
            self.scheduler_host,
            task_port=ports["task_port"],
            shuffle_port=ports["shuffle_port"],
        )
        _write_config(self._config_path, toml)

        logger.info("Starting %s on %s", self.worker_name, self.worker_host)

        args = service_args(self._config_path)
        env: dict[str, str] = {
            **os.environ,
            "POLARS_EULA_ACCEPTED": "yes" if self.config_cluster.accept_eula else "no",
            "POLARS_TEMP_DIR": self.config_cluster.worker.temporary_data_dir,
            "PLC_LOG_LEVEL": self.config_cluster.log_level,
            "RUST_BACKTRACE": self.config_cluster.rust_backtrace,
        }

        self._runner.start(args, env)

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
