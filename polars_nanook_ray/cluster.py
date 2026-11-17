import logging
import time
import typing

import ray
from ray.exceptions import (
    ActorDiedError,
    ActorUnavailableError,
    GetTimeoutError,
    RayError,
)

from polars_nanook_ray.actors import (
    NoRoomForWorkers,
    PolarsScalerActor,
    PolarsSchedulerActor,
    RayWorkers,
    resolve_actor_handles,
    resolve_scaler_name,
    terminate_actors,
)
from polars_nanook_ray.actors.head_node import placement_options
from polars_nanook_ray.actors.pool import WorkerPool
from polars_nanook_ray.config import PolarsRayClusterConfig
from polars_nanook_ray.identity import SCHEDULER

if typing.TYPE_CHECKING:
    from polars_nanook_ray.identity import WorkerId

logger = logging.getLogger(__name__)


class PolarsRayCluster:
    """Manage the lifecycles of the scheduler, workers and scaler actors.

    ```py
    import polars as pl
    import polars_nanook as pn
    import ray

    from polars_nanook_ray.cluster import PolarsRayCluster
    from polars_nanook_ray.config import (
        PolarsObservatoryConfig,
        PolarsRayClusterConfig,
        PolarsSchedulerConfig,
        PolarsServiceAccountLicenseConfig,
        PolarsWorkerConfig,
    )

    config = PolarsRayClusterConfig(
        num_workers=4,
        license=PolarsServiceAccountLicenseConfig(
            workspace_id="<WORKSPACE_ID>",
            client_id="<SERVICE_ACCOUNT_ID>",
            client_secret="<SERVICE_ACCOUNT_SECRET>",
        ),
        scheduler=PolarsSchedulerConfig(...),
        worker=PolarsWorkerConfig(...),
    )

    ray.init(address="auto", namespace=config.cluster_id)
    cluster = PolarsRayCluster(config)
    cluster.start()

    print(
        pl.LazyFrame({"a": [1, 2, 3], "b": [4, 4, 5]})
        .with_columns(pl.col("a").max().over("b").alias("c"))
        .remote(pn.ClusterContext(uri=f"http://{cluster.get_client_addr()}"))
        .execute()
        .head
    )

    cluster.stop()
    ray.shutdown()
    ```
    """

    def __init__(self, config: PolarsRayClusterConfig) -> None:
        self.config = config

        self._scaler_actor: typing.Any = None
        self._scheduler_actor: typing.Any = None
        self._worker_actors: dict[WorkerId, typing.Any] = {}

    def _actor_exists(self, name: str) -> bool:
        try:
            ray.get_actor(name, namespace=self.config.cluster_id)
        except ValueError:
            return False
        return True

    def _scheduler_options(self) -> dict[str, typing.Any]:
        return placement_options(
            "scheduler",
            resource=self.config.scheduler.placement_resource,
            num_cpus=self.config.scheduler.cpus_hint,
            memory=self.config.scheduler.memory_hint,
            single_node=self.config.scheduler.scaling.enabled,
        )

    def _start_scheduler(self) -> None:
        try:
            self._scheduler_actor = ray.get_actor(
                SCHEDULER.actor_name,
                namespace=self.config.cluster_id,
            )
        except ValueError:
            pass
        else:
            logger.info("Reconnected to existing scheduler actor")
            return

        self._scheduler_actor = PolarsSchedulerActor.options(  # type: ignore[attr-defined]
            name=SCHEDULER.actor_name,
            namespace=self.config.cluster_id,
            lifetime="detached",
            max_restarts=self.config.scheduler_max_restarts,
            **self._scheduler_options(),
        ).remote(self.config)

    def _ensure_workers(self) -> None:
        workers = ray.get(
            self._scheduler_actor.ensure_workers.remote(self.config.num_workers),
            timeout=self.config.actor_response_timeout,
        )
        self._worker_actors = resolve_actor_handles(workers, self.config.cluster_id)

    def _start_scaler(self) -> None:
        actor_name = resolve_scaler_name()

        try:
            self._scaler_actor = ray.get_actor(
                actor_name,
                namespace=self.config.cluster_id,
            )
        except ValueError:
            pass
        else:
            logger.info("Reconnected to existing scaler actor")
            return

        self._scaler_actor = PolarsScalerActor.options(  # type: ignore[attr-defined]
            name=actor_name,
            namespace=self.config.cluster_id,
            lifetime="detached",
            max_restarts=self.config.scheduler_max_restarts,
            **placement_options(  # same node as the scheduler
                "scaler",
                resource=self.config.scheduler.placement_resource,
                num_cpus=0,
                single_node=True,
            ),
        ).remote(self.config)

    def _wait_for_scheduler(self) -> None:
        logger.info(
            "Waiting up to %ds for the scheduler",
            self.config.startup_timeout,
        )

        deadline = time.monotonic() + self.config.startup_timeout
        while (remaining := deadline - time.monotonic()) > 0:
            try:
                if ray.get(self._scheduler_actor.is_ready.remote(), timeout=remaining):
                    logger.info("Scheduler started and listening")
                    return
            except GetTimeoutError:
                break
            except ActorUnavailableError:  # Ray is restarting it: not ready yet
                time.sleep(0.5)
                continue
            except ActorDiedError as exc:
                msg = (
                    "The scheduler exited while starting, past its restart limit; "
                    "see the scheduler actor's log for the engine's error"
                    f"\n{exc}"
                )
                raise RuntimeError(msg) from exc
            time.sleep(0.5)

        msg = f"Scheduler did not start within {self.config.startup_timeout}s"
        raise RuntimeError(msg)

    def _wait_for_workers(self) -> None:
        logger.info(
            "Waiting up to %ds for %d workers",
            self.config.startup_timeout,
            len(self._worker_actors),
        )

        pending = dict(self._worker_actors)

        deadline = time.monotonic() + self.config.startup_timeout
        while pending and (remaining := deadline - time.monotonic()) > 0:
            try:
                readiness = ray.get(
                    [actor.is_ready.remote() for actor in pending.values()],
                    timeout=remaining,
                )
            except GetTimeoutError:
                break
            except ActorUnavailableError:  # Ray is restarting it: not ready yet
                time.sleep(0.5)
                continue
            except ActorDiedError as exc:
                msg = (
                    "A worker exited while starting, past its restart limit; see the "
                    "worker actors' logs for the engine's error"
                    f"\n{exc}"
                )
                raise RuntimeError(msg) from exc

            for worker, worker_ready in zip(
                list(pending.keys()),
                readiness,
                strict=True,
            ):
                if worker_ready:
                    logger.info("Worker %s is ready", worker)
                    pending.pop(worker)

            if pending:
                time.sleep(0.5)

        if pending:
            msg = (
                f"Workers {[str(worker) for worker in sorted(pending)]} "
                f"did not start within {self.config.startup_timeout}s; they may be "
                f"waiting for a Ray node with {self.config.worker.cpus_hint:g} CPU and "
                f"{self.config.worker.memory_hint / 1024**3:g} GiB of memory free, "
                "and Ray's autoscaler may be adding nodes: see `ray status`."
            )
            raise RuntimeError(msg)

    def _wait_for_scaler(self) -> None:
        logger.info(
            "Waiting up to %ds for the scaler",
            self.config.startup_timeout,
        )

        deadline = time.monotonic() + self.config.startup_timeout
        while (remaining := deadline - time.monotonic()) > 0:
            try:
                if ray.get(self._scaler_actor.is_ready.remote(), timeout=remaining):
                    logger.info("Scaler actor started and listening")
                    return
            except GetTimeoutError:
                break
            time.sleep(0.5)

        msg = f"Scaler did not start within {self.config.startup_timeout}s"
        raise RuntimeError(msg)

    # this should probably never be called directly
    def _rescale_worker_pool_to(
        self,
        num_workers: int,
        delete: set[str] | None = None,
        keep: set[str] | None = None,
    ) -> None:
        if self._scheduler_actor is None:
            return
        ray.get(
            self._scheduler_actor.rescale_worker_pool_to.remote(
                num_workers,
                delete=delete,
                keep=keep,
            ),
            timeout=self.config.actor_response_timeout,
        )

    def start(self) -> None:
        """Start (or reconnect to) our various actors; block until ready.

        Has the scheduler top the worker pool up to `num_workers`, keeping any worker
        already running, so reconnecting never adds workers. Fails right away if a
        worker's `cpus_hint` and `memory_hint` exceed those of every alive Ray node,
        removing the actors it started; workers that fit a busy node wait for Ray to
        have room for them.
        """
        created = not self._actor_exists(SCHEDULER.actor_name)
        if created:
            self._scheduler_options()  # fail before the scaler is created

        if self.config.scheduler.scaling.enabled:
            self._start_scaler()
            self._wait_for_scaler()

        self._start_scheduler()
        try:
            self._ensure_workers()
        except NoRoomForWorkers as exc:
            if created:
                self.stop()
            raise RuntimeError(str(getattr(exc, "cause", exc))) from None

        self._wait_for_scheduler()
        self._wait_for_workers()

        msg = (
            f"Engine cluster ready: scheduler at {self.get_client_addr()}, "
            f"{len(self._worker_actors)} workers."
        )
        if self.config.scheduler.observatory.enabled:
            msg += f" Dashboard available at {self.get_dashboard_addr()}."
        logger.info(msg)

    def stop(self) -> None:
        """Terminate all the actors (scaler, workers, scheduler)."""
        try:
            self._scheduler_actor = ray.get_actor(
                SCHEDULER.actor_name,
                namespace=self.config.cluster_id,
            )
        except ValueError:
            self._scheduler_actor = None

        if self.config.scheduler.scaling.enabled:
            try:
                self._scaler_actor = ray.get_actor(
                    resolve_scaler_name(),
                    namespace=self.config.cluster_id,
                )
            except ValueError:
                self._scaler_actor = None

            if self._scaler_actor is not None:
                logger.info("Stopping scaler...")
                terminate_actors(
                    [self._scaler_actor],
                    self.config.actor_response_timeout,
                )
                self._scaler_actor = None

        self._stop_workers()
        self._worker_actors = {}

        if self._scheduler_actor is not None:
            logger.info("Stopping scheduler...")
            terminate_actors(
                [self._scheduler_actor],
                self.config.actor_response_timeout,
            )
            self._scheduler_actor = None

        logger.info("Cluster stopped")

    def _stop_workers(self) -> None:
        if self._scheduler_actor is not None:
            try:
                ray.get(
                    self._scheduler_actor.stop_workers.remote(),
                    # it stops the workers with the same timeout
                    timeout=2 * self.config.actor_response_timeout,
                )
            except RayError:
                logger.exception("The scheduler could not stop the workers")
            else:
                return

        # no scheduler to stop them: stop any worker left behind from here
        WorkerPool(RayWorkers(self.config), self.config, 0).stop()

    def get_client_addr(self) -> str:
        """Return the URI queries should be submitted to."""
        if self._scheduler_actor is None:
            msg = "The scheduler does not appear to be running."
            raise RuntimeError(msg)

        return str(
            ray.get(
                self._scheduler_actor.get_client_addr.remote(),
                timeout=self.config.actor_response_timeout,
            )
        )

    def get_dashboard_addr(self) -> str:
        """Return the observatory dashboard URL."""
        if self._scheduler_actor is None:
            msg = "The scheduler does not appear to be running."
            raise RuntimeError(msg)

        if not self.config.scheduler.observatory.enabled:
            return "The observatory is not enabled on this cluster"

        return str(
            ray.get(
                self._scheduler_actor.get_dashboard_addr.remote(),
                timeout=self.config.actor_response_timeout,
            )
        )
