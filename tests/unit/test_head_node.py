import time
import typing

import pytest
import ray
from ray.cluster_utils import Cluster

from polars_nanook_ray.actors.head_node import head_node_options
from polars_nanook_ray.cluster import PolarsRayCluster
from polars_nanook_ray.config import (
    PolarsEnterpriseLicenseConfig,
    PolarsRayClusterConfig,
    PolarsScalingConfig,
    PolarsSchedulerConfig,
    PolarsWorkerConfig,
)


@ray.remote
class _NodeProbe:
    def node_id(self) -> str:
        return ray.get_runtime_context().get_node_id()


@pytest.fixture(scope="module")
def head_node_id() -> typing.Iterator[str]:
    """A two-node cluster whose head has no CPUs, as Ray recommends."""
    cluster = Cluster(initialize_head=True, head_node_args={"num_cpus": 0})
    cluster.add_node(num_cpus=2)
    ray.init(address=cluster.address)
    cluster.wait_for_nodes()

    assert cluster.head_node is not None
    yield cluster.head_node.node_id

    ray.shutdown()
    cluster.shutdown()


def _cluster_config(**scheduler: typing.Any) -> PolarsRayClusterConfig:
    return PolarsRayClusterConfig(
        cluster_id="head-node-test",
        license=PolarsEnterpriseLicenseConfig(license_path="/dev/null"),
        scheduler=PolarsSchedulerConfig(**scheduler),
        worker=PolarsWorkerConfig(),
    )


def test_actor_lands_on_cpu_less_head(head_node_id: str) -> None:
    probe = _NodeProbe.options(**head_node_options("probe", num_cpus=0)).remote()  # type: ignore[attr-defined]

    assert ray.get(probe.node_id.remote(), timeout=30) == head_node_id
    ray.kill(probe)


def test_default_scheduler_fits_cpu_less_head(head_node_id: str) -> None:
    scheduler = PolarsSchedulerConfig()
    options = head_node_options(
        "scheduler",
        num_cpus=scheduler.cpus_hint,
        memory=scheduler.memory_hint,
    )
    probe = _NodeProbe.options(**options).remote()  # type: ignore[attr-defined]

    assert ray.get(probe.node_id.remote(), timeout=30) == head_node_id
    ray.kill(probe)


@pytest.mark.usefixtures("head_node_id")
def test_infeasible_request_names_the_actor_and_resource() -> None:
    with pytest.raises(RuntimeError, match=r"scheduler.*1 CPU.*head node has 0"):
        head_node_options("scheduler", num_cpus=1)


@pytest.mark.usefixtures("head_node_id")
@pytest.mark.parametrize("scaling", [False, True])
def test_cluster_fails_fast_without_creating_actors(scaling: bool) -> None:
    config = _cluster_config(
        cpus_hint=1,
        scaling=PolarsScalingConfig(enabled=scaling),
    )

    started = time.monotonic()
    with pytest.raises(RuntimeError, match="scheduler"):
        PolarsRayCluster(config).start()

    assert time.monotonic() - started < config.startup_timeout / 2
    # from the GCS: the State API reads the dashboard of the newest cluster on the host
    assert not [
        actor
        for actor in ray.util.list_named_actors(all_namespaces=True)
        if actor["namespace"] == config.cluster_id
    ]
