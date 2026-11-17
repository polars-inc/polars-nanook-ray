import logging
import time
import typing
import urllib.parse

import ray
from ray.exceptions import ActorDiedError, ActorUnavailableError, GetTimeoutError

from polars_nanook_ray.actors import (
    Endpoints,
    NoRoomForWorkers,
    PolarsScalerActor,
    PolarsSchedulerActor,
    RayWorkers,
    WorkerExited,
    resolve_scaler_name,
    terminate_actors,
)
from polars_nanook_ray.actors.head_node import placement_options
from polars_nanook_ray.config import PolarsRayClusterConfig
from polars_nanook_ray.identity import SCHEDULER

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
        # workers_per_node=4,
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

    def _find_actor(self, name: str) -> typing.Any:
        try:
            return ray.get_actor(name, namespace=self.config.cluster_id)
        except ValueError:
            return None

    def _find_or_create_actor(
        self,
        name: str,
        actor_class: typing.Any,
        options: typing.Callable[[], dict[str, typing.Any]],
    ) -> typing.Any:
        """Return the actor `name`, creating it with the placement `options` if none."""
        if (actor := self._find_actor(name)) is not None:
            logger.info("Reconnected to existing %s actor", name)
            return actor

        return actor_class.options(
            name=name,
            namespace=self.config.cluster_id,
            lifetime="detached",
            max_restarts=self.config.scheduler_max_restarts,
            **options(),
        ).remote(self.config)

    def _scheduler_options(self) -> dict[str, typing.Any]:
        return placement_options(
            "scheduler",
            resource=self.config.scheduler.placement_resource,
            num_cpus=self.config.scheduler.cpus_hint,
            memory=self.config.scheduler.memory_hint,
            single_node=self.config.scheduler.scaling.enabled,
        )

    def _scaler_options(self) -> dict[str, typing.Any]:
        return placement_options(  # same node as the scheduler
            "scaler",
            resource=self.config.scheduler.placement_resource,
            num_cpus=0,
            single_node=True,
        )

    def _wait_until_ready(self, actor: typing.Any, name: str) -> None:
        """Wait until `actor` has nothing left starting: itself, or what it runs."""
        timeout = self.config.startup_timeout
        logger.info("Waiting up to %ds for the %s", timeout, name)

        not_ready = [name]
        deadline = time.monotonic() + timeout
        while (remaining := deadline - time.monotonic()) > 0:
            try:
                not_ready = ray.get(actor.not_ready.remote(), timeout=remaining)
            except GetTimeoutError:
                break
            except ActorUnavailableError:  # Ray is restarting it: not ready yet
                pass
            except ActorDiedError as exc:
                msg = (
                    f"The {name} exited while starting, past its restart limit; see "
                    f"the {name} actor's log for the error\n{exc}"
                )
                raise RuntimeError(msg) from exc
            except WorkerExited as exc:
                # Ray raises a subclass of the actor's error, with the actor's traceback
                raise exc.cause from None  # type: ignore[attr-defined]
            else:
                if not not_ready:
                    logger.info("The %s is ready", name)
                    return
            time.sleep(0.5)

        msg = f"Did not start within {timeout}s: {', '.join(not_ready)}"
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
        already running, so reconnecting never adds workers. Fails right away if the
        nodes have no room for `num_workers`, removing the actors it started.
        """
        created = self._find_actor(SCHEDULER.actor_name) is None
        if created:
            self._scheduler_options()  # fail before the scaler is created

        if self.config.scheduler.scaling.enabled:
            self._scaler_actor = self._find_or_create_actor(
                resolve_scaler_name(), PolarsScalerActor, self._scaler_options
            )
            self._wait_until_ready(self._scaler_actor, "scaler")

        self._scheduler_actor = self._find_or_create_actor(
            SCHEDULER.actor_name, PolarsSchedulerActor, self._scheduler_options
        )
        try:
            workers = ray.get(
                self._scheduler_actor.ensure_workers.remote(self.config.num_workers),
                timeout=self.config.actor_response_timeout,
            )
        except NoRoomForWorkers as exc:
            if created:
                self.stop()
            # Ray raises a subclass of the actor's error, with the actor's traceback
            raise exc.cause from None  # type: ignore[attr-defined]

        self._wait_until_ready(self._scheduler_actor, "scheduler")

        endpoints = self._endpoints()
        msg = f"Engine cluster ready: scheduler at {endpoints.client}, "
        msg += f"{len(workers)} workers."
        if endpoints.dashboard is not None:
            msg += f" Dashboard available at {endpoints.dashboard}."
        logger.info(msg)

    def stop(self) -> None:
        """Terminate all the actors (scaler, workers, scheduler)."""
        if self.config.scheduler.scaling.enabled:
            self._scaler_actor = self._find_actor(resolve_scaler_name())
            if self._scaler_actor is not None:
                logger.info("Stopping scaler...")
                terminate_actors(
                    [self._scaler_actor],
                    self.config.actor_response_timeout,
                )
                self._scaler_actor = None

        self._scheduler_actor = self._find_actor(SCHEDULER.actor_name)
        if self._scheduler_actor is not None:
            logger.info("Stopping workers and scheduler...")
            terminate_actors(
                [self._scheduler_actor],
                self.config.actor_response_timeout,
            )
            self._scheduler_actor = None

        # any worker left behind by a scheduler that is gone or failed to stop it
        workers = RayWorkers(self.config)
        workers.terminate(workers.running())

        logger.info("Cluster stopped")

    def _endpoints(self) -> Endpoints:
        if self._scheduler_actor is None:
            msg = "The scheduler does not appear to be running."
            raise RuntimeError(msg)

        return ray.get(  # type: ignore[no-any-return]
            self._scheduler_actor.endpoints.remote(),
            timeout=self.config.actor_response_timeout,
        )

    def get_client_addr(self) -> str:
        """Return the `host:port` queries should be submitted to."""
        return urllib.parse.urlsplit(self._endpoints().client).netloc

    def get_dashboard_addr(self) -> str:
        """Return the observatory dashboard URL."""
        dashboard = self._endpoints().dashboard
        return dashboard or "The observatory is not enabled on this cluster"
