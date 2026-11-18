import pathlib
import tempfile
import time
import typing
import uuid

import ray
from ray.actor import ActorHandle
from ray.cluster_utils import Cluster

from polars_nanook_ray.actors import (
    PolarsSchedulerActor,
    resolve_actor_handles,
    terminate_actors,
)
from polars_nanook_ray.actors.head_node import placement_options
from polars_nanook_ray.cluster import PolarsRayCluster
from polars_nanook_ray.config import (
    PolarsEnterpriseLicenseConfig,
    PolarsObservatoryConfig,
    PolarsRayClusterConfig,
    PolarsSchedulerConfig,
    PolarsWorkerConfig,
)
from polars_nanook_ray.identity import SCHEDULER, WorkerId, list_workers


def head_cluster() -> Cluster:
    """Start a Ray cluster with a head without CPUs, as Ray recommends, and join it."""
    cluster = Cluster(
        initialize_head=True,
        head_node_args={"num_cpus": 0, "include_dashboard": False},
    )
    ray.init(address=cluster.address)
    return cluster


_STUB_ENGINE = pathlib.Path(tempfile.gettempdir()) / "polars-nanook-ray-stub-engine"
_STUB_ENGINE.write_text("#!/bin/sh\nexec sleep 3600\n")
_STUB_ENGINE.chmod(0o755)


def make_config(worker_cpus: int = 1, **kwargs: typing.Any) -> PolarsRayClusterConfig:
    return PolarsRayClusterConfig(
        # an engine that keeps running, or its actor exits with it
        binary_path=str(_STUB_ENGINE),
        cluster_id=f"polars-nanook-{uuid.uuid4().hex[:8]}",
        license=PolarsEnterpriseLicenseConfig(license_path="/nonexistent.json"),
        scheduler=PolarsSchedulerConfig(
            cpus_hint=0,
            memory_hint=1,
            observatory=PolarsObservatoryConfig(enabled=False),
        ),
        worker=PolarsWorkerConfig(cpus_hint=worker_cpus, memory_hint=1),
        **kwargs,
    )


def run_scheduler(config: PolarsRayClusterConfig) -> typing.Iterator[ActorHandle]:
    """Yield a scheduler actor on the head node, then stop it and its workers."""
    actor = PolarsSchedulerActor.options(  # type: ignore[attr-defined]
        name=SCHEDULER.actor_name,
        namespace=config.cluster_id,
        **placement_options(
            "scheduler",
            resource=config.scheduler.placement_resource,
            num_cpus=config.scheduler.cpus_hint,
            memory=config.scheduler.memory_hint,
        ),
    ).remote(config)

    yield actor

    PolarsRayCluster(config).stop()
    terminate_actors([actor], config.actor_response_timeout)


def wait_for_workers(cluster_id: str, expected: int) -> dict[WorkerId, ActorHandle]:
    deadline = time.monotonic() + 30
    workers = list_workers(cluster_id)
    while len(workers) != expected and time.monotonic() < deadline:
        time.sleep(0.5)
        workers = list_workers(cluster_id)

    assert len(workers) == expected, sorted(workers)
    return resolve_actor_handles(workers, cluster_id)


def node_id(actor: ActorHandle) -> str:
    return str(
        ray.get(
            actor.__ray_call__.remote(lambda _: ray.get_runtime_context().get_node_id())
        )
    )
