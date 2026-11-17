import time
import typing

import pytest
import ray
from ray.cluster_utils import Cluster

from polars_nanook_ray.actors.head_node import placement_options
from polars_nanook_ray.cluster import PolarsRayCluster
from polars_nanook_ray.config import (
    HEAD_NODE_RESOURCE,
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


class _Nodes(typing.NamedTuple):
    head: str
    labeled: str


@pytest.fixture(scope="module")
def nodes() -> typing.Iterator[_Nodes]:
    """A CPU-less head, as Ray recommends, and a node with the custom resource."""
    cluster = Cluster(initialize_head=True, head_node_args={"num_cpus": 0})
    labeled = cluster.add_node(num_cpus=2, resources={"scheduler_node": 1})
    ray.init(address=cluster.address)
    cluster.wait_for_nodes()

    assert cluster.head_node is not None
    yield _Nodes(head=cluster.head_node.node_id, labeled=labeled.node_id)

    ray.shutdown()
    cluster.shutdown()


def _node_of(options: dict[str, typing.Any]) -> str:
    probe = _NodeProbe.options(**options).remote()  # type: ignore[attr-defined]
    try:
        return ray.get(probe.node_id.remote(), timeout=30)
    finally:
        ray.kill(probe)


def _cluster_config(**scheduler: typing.Any) -> PolarsRayClusterConfig:
    return PolarsRayClusterConfig(
        cluster_id="head-node-test",
        license=PolarsEnterpriseLicenseConfig(license_path="/dev/null"),
        scheduler=PolarsSchedulerConfig(**scheduler),
        worker=PolarsWorkerConfig(),
    )


def test_default_resource_is_the_head_node() -> None:
    assert PolarsSchedulerConfig().placement_resource == HEAD_NODE_RESOURCE


def test_actor_lands_on_cpu_less_head(nodes: _Nodes) -> None:
    options = placement_options("probe", resource=HEAD_NODE_RESOURCE, num_cpus=0)

    assert _node_of(options) == nodes.head


def test_default_scheduler_fits_cpu_less_head(nodes: _Nodes) -> None:
    scheduler = PolarsSchedulerConfig()
    options = placement_options(
        "scheduler",
        resource=scheduler.placement_resource,
        num_cpus=scheduler.cpus_hint,
        memory=scheduler.memory_hint,
    )

    assert _node_of(options) == nodes.head


@pytest.mark.usefixtures("nodes")
def test_infeasible_request_names_the_actor_and_resource() -> None:
    with pytest.raises(
        RuntimeError,
        match=r"scheduler on the Ray head node \(`node:__internal_head__`\).*"
        r"1 CPU.*has 0 CPU",
    ):
        placement_options("scheduler", resource=HEAD_NODE_RESOURCE, num_cpus=1)


def test_custom_resource_pins_to_the_node_advertising_it(nodes: _Nodes) -> None:
    options = placement_options("probe", resource="scheduler_node", num_cpus=0)

    assert options["resources"] == {"scheduler_node": 0.001}
    assert _node_of(options) == nodes.labeled


@pytest.mark.usefixtures("nodes")
def test_custom_resource_too_small_names_the_resource() -> None:
    with pytest.raises(RuntimeError, match=r"`scheduler_node`.*3 CPU.*has 2 CPU"):
        placement_options("scheduler", resource="scheduler_node", num_cpus=3)


@pytest.mark.usefixtures("nodes")
def test_missing_custom_resource_fails_fast() -> None:
    with pytest.raises(
        RuntimeError,
        match=r"scheduler: no alive Ray node advertises the `no_such_node` resource",
    ):
        placement_options("scheduler", resource="no_such_node", num_cpus=0)


def test_no_resource_leaves_placement_to_ray(nodes: _Nodes) -> None:
    options = placement_options("probe", resource=None, num_cpus=1)

    assert "resources" not in options
    # the only node with a CPU to give
    assert _node_of(options) == nodes.labeled


@pytest.mark.usefixtures("nodes")
def test_no_resource_still_fails_fast_when_no_node_fits() -> None:
    with pytest.raises(RuntimeError, match=r"scheduler on any Ray node.*3 CPU"):
        placement_options("scheduler", resource=None, num_cpus=3)


@pytest.mark.usefixtures("nodes")
def test_scaler_cannot_be_split_from_the_scheduler() -> None:
    with pytest.raises(RuntimeError, match=r"2 alive Ray nodes.*scaler"):
        placement_options("scheduler", resource=None, num_cpus=0, single_node=True)


@pytest.mark.usefixtures("nodes")
@pytest.mark.parametrize("scaling", [False, True])
@pytest.mark.parametrize(
    ("scheduler", "error"),
    [
        ({"cpus_hint": 1}, "scheduler.*1 CPU"),
        ({"placement_resource": "no_such_node"}, "scheduler.*`no_such_node`"),
    ],
)
def test_cluster_fails_fast_without_creating_actors(
    scaling: bool, scheduler: dict[str, typing.Any], error: str
) -> None:
    config = _cluster_config(
        **scheduler,
        scaling=PolarsScalingConfig(enabled=scaling),
    )

    started = time.monotonic()
    with pytest.raises(RuntimeError, match=error):
        PolarsRayCluster(config).start()

    assert time.monotonic() - started < config.startup_timeout / 2
    # from the GCS: the State API reads the dashboard of the newest cluster on the host
    assert not [
        actor
        for actor in ray.util.list_named_actors(all_namespaces=True)
        if actor["namespace"] == config.cluster_id
    ]
