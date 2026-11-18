import logging
import os
import time
import typing

import ray
from polars_nanook import ClientOptions, ClusterContext
from ray.exceptions import ActorDiedError, ActorUnavailableError, GetTimeoutError

from polars_nanook_ray._spec import HEAD_NODE_RESOURCE, build_spec
from polars_nanook_ray.actors import (
    Endpoints,
    NoRoomForWorkers,
    PolarsSchedulerActor,
    WorkerExited,
    terminate_actors,
)
from polars_nanook_ray.actors.head_node import scheduler_options
from polars_nanook_ray.actors.utils import ACTOR_RESPONSE_TIMEOUT, creation_error
from polars_nanook_ray.identity import SCHEDULER

if typing.TYPE_CHECKING:
    from typing import Self

    from polars_nanook_ray._spec import ClusterSpec, StorageLike
    from polars_nanook_ray.options import Autoscale, Ports, ServiceAccount, Tls

logger = logging.getLogger(__name__)

# past `startup_timeout`, for the actors to report why their engine did not start
STARTUP_GRACE = 10.0


class RayClusterContext(ClusterContext):
    """A Polars Nanook cluster on Ray, and the context to run queries on it.

    Constructing it validates the keywords only; `start()`, or entering it as a
    context manager, starts the cluster on the Ray cluster `ray.init` connected to, and
    `stop()` stops it.

    ```py
    import polars as pl
    import ray

    from polars_nanook_ray import RayClusterContext

    ray.init(address="auto")
    with RayClusterContext(license="/etc/polars/license.json", workers=4) as ctx:
        pl.LazyFrame({"a": [1, 2, 3]}).remote(ctx).execute()
    ```

    Parameters
    ----------
    license
        A license file path, a license server `http(s)://` URI, or a `ServiceAccount`
        (or the mapping of one) validating online.
    name
        The cluster ID, also the Ray namespace of its actors.
    workers
        The number of workers to start.
    cpus
        The CPUs of each worker: what it requests from Ray, and reserves in the engine.
        Ray spreads the workers over the nodes with these CPUs and `memory` free.
    memory
        The memory of each worker, in GiB, as requested from Ray.
    autoscale
        Let the engine scale the workers to the queries, within the bounds of an
        `Autoscale` (or the mapping of one); `None` keeps `workers` fixed.
    shuffle_data
        Where workers exchange shuffle data: a `Storage`, URL or absolute path.
    anonymous_results
        Where the results of anonymous queries go.
    spill_dir
        The local directory Polars spills to on each worker node.
    ports
        The ports to listen on; those unset are picked free on each node. Ray may
        place two workers on one node, so a fixed `task` or `shuffle` port makes the
        second fail to start: leave those unset.
    tls
        Serve the client port and the dashboard over TLS, a `Tls` (or the mapping of
        one); its `auth` also requires a JWT on them.
    worker_max_restarts
        How many times Ray restarts a worker, also on another node when its node is
        lost (-1 = unlimited).
    scheduler_max_restarts
        How many times Ray restarts the scheduler (-1 = unlimited).
    reconcile_interval
        Seconds between checks that start workers up to the target, replacing those
        past their restart limit; 0 disables them.
    startup_timeout
        Seconds the scheduler and each worker have for their engine to accept
        connections; past it, or if the engine exits first, the actor fails without
        restarts, and `start()` raises with the errors the engine logged.
    scheduler_resource
        The Ray resource of the node to run the scheduler on, the head node by
        default; `None` places it anywhere.
    binary_path
        The engine binary; by default, that of the `polars-nanook-compute` package.
    env_vars
        Environment variables of every engine process.
    log_level
        The engine's log level.
    observatory
        Run the observatory and its dashboard.
    scheduler_overrides
        Deep-merged into the scheduler's engine TOML; `None` deletes a key. Engine
        settings without a keyword go here: `{"scheduler": {"deny_anonymous_users":
        True}}` rejects anonymous queries, `{"scheduler": {"allow_local_scans": False,
        "allow_local_sinks": False}}` forbids I/O on the nodes' local disks,
        `{"monitoring": {"host_metrics": {"enabled": False}}}` (in both overrides)
        stops host metrics, and `{"lineage": {"transport": {"stdio": {"output_stream":
        "stdout"}}}}` (in both overrides) emits lineage events.
    worker_overrides
        Deep-merged into each worker's engine TOML; `None` deletes a key. E.g.
        `{"memory_limit": 2 * 1024**3}` caps each worker engine to 2 GiB, through
        cgroups.

    Raises
    ------
    ValueError
        If a keyword is invalid, naming it.
    """

    def __init__(
        self,
        *,
        license: "str | os.PathLike[str] | ServiceAccount | typing.Mapping[str, typing.Any]",
        name: str = "polars-nanook",
        workers: int = 1,
        cpus: float = 1,
        memory: float = 2,
        autoscale: "Autoscale | typing.Mapping[str, int | None] | None" = None,
        shuffle_data: "StorageLike" = "/tmp/polars/shuffle-data",
        anonymous_results: "StorageLike" = "/tmp/polars/anonymous-results",
        spill_dir: str | os.PathLike[str] = "/tmp/polars/temporary-data",
        ports: "Ports | typing.Mapping[str, int | None] | None" = None,
        tls: "Tls | typing.Mapping[str, typing.Any] | None" = None,
        worker_max_restarts: int = 3,
        scheduler_max_restarts: int = 3,
        reconcile_interval: float = 10.0,
        startup_timeout: float = 30.0,
        scheduler_resource: str | None = HEAD_NODE_RESOURCE,
        binary_path: str | os.PathLike[str] | None = None,
        env_vars: typing.Mapping[str, str] | None = None,
        log_level: typing.Literal["info", "debug", "trace"] = "info",
        observatory: bool = True,
        scheduler_overrides: typing.Mapping[str, typing.Any] | None = None,
        worker_overrides: typing.Mapping[str, typing.Any] | None = None,
    ) -> None:
        keywords = {k: v for k, v in locals().items() if k != "self"}
        self._spec: ClusterSpec = build_spec(**keywords)
        self._scheduler_actor: typing.Any = None

    @classmethod
    def connect(cls, name: str = "polars-nanook") -> "Self":
        """Attach to the running cluster `name`, as started from another driver.

        Raises `ValueError` if no cluster of that name runs.
        """
        try:
            actor = ray.get_actor(SCHEDULER.actor_name, namespace=name)
        except ValueError:
            msg = f"No cluster named {name!r} is running"
            raise ValueError(msg) from None
        ctx = cls.__new__(cls)
        ctx._spec = typing.cast(
            "ClusterSpec",
            ray.get(actor.spec.remote(), timeout=ACTOR_RESPONSE_TIMEOUT),
        )
        ctx._scheduler_actor = actor
        ctx._connect_client()
        return ctx

    def start(self) -> None:
        """Start (or reconnect to) the cluster; block until ready.

        Waits for the scheduler's engine, then has the scheduler top the workers up to
        `workers`, keeping any already running, so reconnecting never adds workers.
        Fails right away if a worker's `cpus` and `memory` exceed those of every alive
        Ray node, and with the engine's error if an engine exits while starting; workers
        that fit a busy node wait for Ray to have room for them. On failure, removes the
        actors it started.
        """
        created = self._find_scheduler() is None
        self._scheduler_actor = self._find_scheduler() or self._create_scheduler()
        try:
            self._wait_until_ready()
            try:
                workers = ray.get(
                    self._scheduler_actor.ensure_workers.remote(self._spec.workers),
                    timeout=ACTOR_RESPONSE_TIMEOUT,
                )
            except NoRoomForWorkers as exc:
                # Ray raises a subclass of the actor's error, with its traceback
                raise exc.cause from None  # type: ignore[attr-defined]
            self._wait_until_ready()
        except Exception:
            if created:
                self.stop()
            raise
        self._connect_client()

        msg = f"Engine cluster ready: scheduler at {self.client_addr}, "
        msg += f"{len(workers)} workers."
        if (dashboard := self.dashboard_addr) is not None:
            msg += f" Dashboard available at {dashboard}."
        logger.info(msg)

    def stop(self) -> None:
        """Stop the workers and the scheduler, and drop the client connection."""
        # without a scheduler there are no workers: they die with it
        if (actor := self._find_scheduler()) is not None:
            logger.info("Stopping workers and scheduler...")
            terminate_actors([actor], ACTOR_RESPONSE_TIMEOUT)
        self._scheduler_actor = None
        for attr in ("_direct_client", "_compute_id", "_connection_mode"):
            self.__dict__.pop(attr, None)
        logger.info("Cluster stopped")

    def __enter__(self) -> "Self":
        self.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.stop()

    @property
    def client_addr(self) -> str:
        """The URL queries are submitted to; raises if the cluster is not running."""
        return self._endpoints().client

    @property
    def dashboard_addr(self) -> str | None:
        """The URL of the observatory dashboard, `None` without the observatory."""
        return self._endpoints().dashboard

    def _find_scheduler(self) -> typing.Any:
        try:
            return ray.get_actor(SCHEDULER.actor_name, namespace=self._spec.name)
        except ValueError:
            return None

    def _create_scheduler(self) -> typing.Any:
        options = scheduler_options(self._spec.scheduler_resource)
        return PolarsSchedulerActor.options(  # type: ignore[attr-defined]
            name=SCHEDULER.actor_name,
            namespace=self._spec.name,
            lifetime="detached",
            max_restarts=self._spec.scheduler_max_restarts,
            **options,
        ).remote(self._spec)

    def _wait_until_ready(self) -> None:
        """Wait until the scheduler and the workers it starts are ready."""
        timeout = self._spec.startup_timeout + STARTUP_GRACE
        logger.info("Waiting up to %gs for the scheduler", timeout)

        not_ready = ["scheduler"]
        deadline = time.monotonic() + timeout
        while (remaining := deadline - time.monotonic()) > 0:
            try:
                not_ready = ray.get(
                    self._scheduler_actor.not_ready.remote(), timeout=remaining
                )
            except GetTimeoutError:
                break
            except ActorUnavailableError:  # Ray is restarting it: not ready yet
                pass
            except ActorDiedError as exc:
                if (error := creation_error(exc)) is not None:
                    raise error from None
                msg = (
                    "The scheduler exited while starting, past its restart limit; see "
                    f"the scheduler actor's log for the error\n{exc}"
                )
                raise RuntimeError(msg) from exc
            except WorkerExited as exc:
                # Ray raises a subclass of the actor's error, with the actor's traceback
                raise exc.cause from None  # type: ignore[attr-defined]
            else:
                if not not_ready:
                    logger.info("The scheduler is ready")
                    return
            time.sleep(0.5)

        msg = f"Did not start within {timeout:g}s: {', '.join(not_ready)}"
        if not_ready != [str(SCHEDULER)]:
            msg += (
                f". Workers not ready may be waiting for a Ray node with {self._spec.cpus:g}"
                f" CPU and {self._spec.memory:g} GiB of memory free; Ray's autoscaler may "
                "be adding nodes, see `ray status`."
            )
        raise RuntimeError(msg)

    def _endpoints(self) -> Endpoints:
        if self._scheduler_actor is None:
            msg = f"The cluster {self._spec.name!r} is not running; call start()"
            raise RuntimeError(msg)
        return ray.get(  # type: ignore[no-any-return]
            self._scheduler_actor.endpoints.remote(), timeout=ACTOR_RESPONSE_TIMEOUT
        )

    def _connect_client(self) -> None:
        endpoints = self._endpoints()
        bootstrap = ClusterContext(
            uri=endpoints.client,
            observatory=(
                None
                if endpoints.dashboard is None
                else ClientOptions(uri=endpoints.dashboard)
            ),
        )
        self._direct_client = bootstrap._direct_client
        self._compute_id = bootstrap._compute_id
        self._connection_mode = bootstrap._connection_mode
