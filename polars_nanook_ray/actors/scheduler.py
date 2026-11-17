import logging
import os
import threading
import time
import typing

import ray

from polars_nanook_ray.actors.placement import (
    WorkerPoolReconciler,
    WorkerSlot,
    plan_workers,
    warn_no_room,
)
from polars_nanook_ray.actors.utils import (
    _handle_sigterm,
    _remove_config,
    _resolve_config_path,
    _write_config,
    resolve_actor_handles,
    terminate_actors,
)
from polars_nanook_ray.actors.worker import start_worker
from polars_nanook_ray.config import PolarsRayClusterConfig
from polars_nanook_ray.identity import SCHEDULER, WorkerId, identify, list_workers
from polars_nanook_ray.ports import resolve_ports
from polars_nanook_ray.subprocess_runner import SubprocessRunner

if typing.TYPE_CHECKING:
    import pathlib


logger = logging.getLogger(__name__)


def _live_workers_of(
    requested: set[str] | None, live: set[WorkerId]
) -> set[WorkerId] | None:
    """Return the live workers the engine instance IDs or actor names given are."""
    return live & {identify(value) for value in requested or ()} or None


@ray.remote
class PolarsSchedulerActor:
    """The central service, distributing tasks and piloting autoscaling."""

    def __init__(self, config: PolarsRayClusterConfig) -> None:
        self.config_cluster: PolarsRayClusterConfig = config

        self.num_workers: int = config.num_workers  # initial value

        self._config_path: pathlib.Path = _resolve_config_path(SCHEDULER)
        self._runner = SubprocessRunner(config.binary_path, SCHEDULER)

        self.scheduler_host: str = self._runner.host

        self._reconciling = threading.Event()
        self.start()
        _handle_sigterm(self.stop)

        if config.worker_reconcile_interval > 0:
            self._reconciler = WorkerPoolReconciler(
                config.worker_reconcile_interval,
                plan=self._plan_workers,
                start=self._start_worker,
                running=self._list_workers,
            )
            threading.Thread(
                target=self._reconcile_workers, name="reconcile-workers", daemon=True
            ).start()

    def _plan_workers(self, target: int, running: set[WorkerId]) -> list[WorkerSlot]:
        return plan_workers(self.config_cluster, target, running)

    def _start_worker(self, slot: WorkerSlot) -> None:
        start_worker(self.config_cluster, slot, self.scheduler_host)

    def _list_workers(self) -> set[WorkerId]:
        return list_workers(self.config_cluster.cluster_id)

    def _reconcile_workers(self) -> None:
        while not self._reconciling.wait(self._reconciler.interval):
            try:
                self._reconciler.reconcile(self.num_workers, time.monotonic())
            except Exception:
                logger.exception("Could not reconcile the worker pool")

    def __ray_shutdown__(self) -> None:
        self.stop()

    def _remove_workers(self, workers: set[WorkerId]) -> None:
        actors = resolve_actor_handles(workers, self.config_cluster.cluster_id)
        terminate_actors(
            list(actors.values()), self.config_cluster.actor_response_timeout
        )

        for worker in workers:
            logger.info("Removed worker %s", worker)

    def start(self) -> None:
        """Spawn the scheduler process, if not already running."""
        if self._runner.is_started():
            return

        self._runner.stop_orphans(self.config_cluster.cluster_id)

        resolve_ports(self.config_cluster.config_bind_ports_scheduler())

        toml = self.config_cluster.config_scheduler(self.scheduler_host)
        _write_config(self._config_path, toml)

        logger.info(
            "Starting scheduler (%s observatory) on %s",
            "and" if self.config_cluster.scheduler.observatory.enabled else "without",
            self.scheduler_host,
        )

        env: dict[str, str] = {
            **os.environ,
            "POLARS_EULA_ACCEPTED": "yes" if self.config_cluster.accept_eula else "no",
            "PLC_LOG_LEVEL": self.config_cluster.log_level,
            "RUST_BACKTRACE": self.config_cluster.rust_backtrace,
        }

        self._runner.start(self._config_path, env)

    def stop_reconciling(self) -> None:
        """Stop refilling the worker pool, before the workers are stopped."""
        self._reconciling.set()

    def stop(self) -> None:
        """Terminate the scheduler process and clean up."""
        self.stop_reconciling()
        self._runner.stop()

        _remove_config(self._config_path)

    def is_ready(self) -> bool:
        """Return whether the scheduler process is alive and accepting registration."""
        self._runner._raise_if_exited()

        return self._runner.is_alive() and self._runner.is_listening(
            self.config_cluster.scheduler.worker_registration_port
        )

    def get_host(self) -> str:
        """Return the IP address the scheduler is running on."""
        return self.scheduler_host

    def get_client_addr(self) -> str:
        """Return the URI queries should be submitted to."""
        host = self.scheduler_host
        port = self.config_cluster.scheduler.client_port
        return f"{host}:{port}"

    def get_dashboard_addr(self) -> str | None:
        """Return the observatory dashboard URL."""
        host = self.scheduler_host
        port = self.config_cluster.scheduler.observatory.rest_port
        return (
            f"http://{host}:{port}"
            if self.config_cluster.scheduler.observatory.enabled
            else None
        )

    def get_scheduler_pid(self) -> int | None:
        """Return the OS process ID of the scheduler binary subprocess."""
        return self._runner.pid()

    def get_scaling_status(self) -> dict[str, int | set[WorkerId] | None]:
        """Return the current/desired worker counts and configured bounds."""
        return {
            "available": self._list_workers(),
            "desired": self.num_workers,
            "min": self.config_cluster.min_workers,
            "max": self.config_cluster.max_workers,
        }

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
            and `max_workers`. If neither `delete` nor `keep` is given, random workers
            are removed.
        delete
            Set of workers to terminate, as engine instance IDs or actor names.
            Applied before `keep`. Values that are not live workers are ignored.
        keep
            Set of workers to keep running, as engine instance IDs or actor names,
            while terminating all the others not already removed via `delete`.
            Values that are not live workers are ignored.

        """
        min_workers = self.config_cluster.min_workers
        max_workers = self.config_cluster.max_workers
        clamped = max(min_workers, num_workers)

        if max_workers is not None:
            clamped = min(max_workers, clamped)

        if clamped != num_workers:
            logger.warning(
                "Clamped requested %d workers to %d, within [%d, %s]",
                num_workers,
                clamped,
                min_workers,
                max_workers,
            )
            num_workers = clamped

        # set first, for the reconciler not to refill the workers removed below
        self.num_workers = num_workers

        workers = self._list_workers()

        _delete = _live_workers_of(delete, workers)
        _keep = _live_workers_of(keep, workers)

        if len(workers) < num_workers:
            slots = self._plan_workers(num_workers, workers)
            if len(workers) + len(slots) < num_workers:
                warn_no_room(len(slots), num_workers - len(workers))
            for slot in slots:
                self._start_worker(slot)

        elif len(workers) > num_workers:
            if num_workers == 0:
                to_remove = set(workers)
            elif _delete is None and _keep is None:
                to_remove = set(list(workers)[: len(workers) - num_workers])
            else:
                to_remove = set()
                remaining = workers
                if _delete is not None:
                    to_remove |= _delete
                    remaining = remaining - _delete
                if _keep is not None:
                    to_remove |= remaining - _keep

            self._remove_workers(to_remove)
