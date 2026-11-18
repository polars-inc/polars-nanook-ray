import os
import shutil
import subprocess
import sysconfig
import typing
import uuid

import polars as pl
import polars_nanook as pn
import pytest
import ray

from polars_nanook_ray import Ports, RayComputeContext
from polars_nanook_ray.identity import WorkerId, list_workers

ContextFactory = typing.Callable[..., RayComputeContext]
TestQuery = typing.Callable[..., pl.DataFrame | None]

RAY_GCS_PORT = 6379
RAY_CLIENT_PORT = 10001


def name_of(ctx: RayComputeContext) -> str:
    return ctx._spec.name


def scheduler_actor(ctx: RayComputeContext) -> typing.Any:
    return ctx._scheduler_actor


def rescale(
    ctx: RayComputeContext,
    num_workers: int,
    delete: set[str] | None = None,
    keep: set[str] | None = None,
) -> None:
    ray.get(
        scheduler_actor(ctx).rescale_worker_pool_to.remote(
            num_workers, delete=delete, keep=keep
        ),
        timeout=30,
    )


def worker_actors(ctx: RayComputeContext) -> dict[WorkerId, typing.Any]:
    """Return the cluster's live workers, listed from the GCS, with their actors."""
    namespace = name_of(ctx)
    return {
        worker: ray.get_actor(worker.actor_name, namespace=namespace)
        for worker in sorted(list_workers(namespace))
    }


def make_context(**keywords: typing.Any) -> RayComputeContext:
    """A context for this machine: free ports, small workers, no observatory."""
    keywords.setdefault("license", os.environ.get("LICENSE_PATH", "./license.json"))
    keywords.setdefault("name", f"polars-nanook-{uuid.uuid4().hex[:8]}")
    keywords.setdefault("binary_path", os.environ.get("BINARY_PATH"))
    # several workers on one host
    keywords.setdefault("cpus", 0)
    keywords.setdefault("memory", 0)
    keywords.setdefault("observatory", False)
    no_host_metrics = {"monitoring": {"host_metrics": {"enabled": False}}}
    keywords.setdefault("scheduler_overrides", no_host_metrics)
    keywords.setdefault("worker_overrides", no_host_metrics)
    keywords.setdefault("ports", Ports(client=None, dashboard=None))
    keywords.setdefault("env_vars", {"POLARS_EULA_ACCEPTED": "yes"})
    return RayComputeContext(**keywords)


def _ld_library_path() -> str:
    libdir = sysconfig.get_config_var("LIBDIR")
    ld_library_path = os.environ.get("LD_LIBRARY_PATH")
    return f"{ld_library_path}:{libdir}" if ld_library_path else libdir


def _python_path() -> str:
    purelib = sysconfig.get_path("purelib")
    python_path = os.environ.get("PYTHONPATH")
    return f"{python_path}:{purelib}" if python_path else purelib


@pytest.fixture(scope="session", autouse=True)
def ray_head() -> typing.Iterator[None]:
    """Start a detached Ray head node for the session."""
    os.environ["LD_LIBRARY_PATH"] = _ld_library_path()
    os.environ["PYTHONPATH"] = _python_path()

    if (binary_path := shutil.which("ray")) is None:
        msg = "Ray is not installed or not in PATH!"
        raise RuntimeError(msg)

    subprocess.run([binary_path, "stop", "--force"], check=False)  # clean slate

    subprocess.run(
        [
            binary_path,
            "start",
            "--head",
            "--dashboard-host=0.0.0.0",
            "--disable-usage-stats",
            f"--port={RAY_GCS_PORT}",
            f"--ray-client-server-port={RAY_CLIENT_PORT}",
        ],
        check=True,
    )

    yield

    subprocess.run([binary_path, "stop", "--force"], check=False)


@pytest.fixture
def ray_cluster() -> typing.Iterator[ContextFactory]:
    """Start clusters from `make_context` keywords; stop them after the test."""
    started: list[RayComputeContext] = []

    def _factory(**keywords: typing.Any) -> RayComputeContext:
        ctx = make_context(**keywords)
        ray.init(address="auto", ignore_reinit_error=True)
        started.append(ctx)
        ctx.start()
        return ctx

    yield _factory

    for ctx in started:
        ctx.stop()

    if ray.is_initialized():
        ray.shutdown()


@pytest.fixture
def run_query() -> TestQuery:
    def _factory(
        ctx: pn.ClusterContext,
        *,
        min_workers: int | None = None,
        max_workers: int | None = None,
    ) -> pl.DataFrame | None:
        query = (
            pl.LazyFrame({"a": [1, 2, 3], "b": [4, 4, 5]})
            .with_columns(pl.col("a").max().over("b").alias("c"))
            .remote(ctx)
        )

        if min_workers is not None or max_workers is not None:
            query = query.distributed(  # type:ignore[assignment]
                min_workers=min_workers,
                max_workers=max_workers,
            )

        return query.execute().head

    return _factory
