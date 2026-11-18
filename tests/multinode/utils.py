import shutil
import time
import typing
import uuid

import ray
from ray.actor import ActorHandle
from ray.cluster_utils import Cluster

from polars_nanook_ray.actors import (
    PolarsSchedulerActor,
    list_worker_names,
    resolve_scheduler_name,
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


def head_cluster() -> Cluster:
    """Start a Ray cluster with a head without CPUs, as Ray recommends, and join it."""
    cluster = Cluster(
        initialize_head=True,
        head_node_args={"num_cpus": 0, "include_dashboard": False},
    )
    ray.init(address=cluster.address)
    return cluster


def make_config(worker_cpus: int = 1, **kwargs: typing.Any) -> PolarsRayClusterConfig:
    binary_path = shutil.which("true")
    assert binary_path is not None

    return PolarsRayClusterConfig(
        binary_path=binary_path,
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
        name=resolve_scheduler_name(),
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


def wait_for_workers(cluster_id: str, expected: int) -> dict[str, ActorHandle]:
    deadline = time.monotonic() + 30
    names = list_worker_names(cluster_id)
    while len(names) != expected and time.monotonic() < deadline:
        time.sleep(0.5)
        names = list_worker_names(cluster_id)

    assert len(names) == expected, sorted(names)
    return {name: ray.get_actor(name, namespace=cluster_id) for name in names}


def node_id(actor: ActorHandle) -> str:
    return str(
        ray.get(
            actor.__ray_call__.remote(lambda _: ray.get_runtime_context().get_node_id())
        )
    )
