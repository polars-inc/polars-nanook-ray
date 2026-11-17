"""Worker placement on a simulated multi-node Ray cluster.

Several raylets run on this one machine, without the dashboard. The binaries are
replaced by `true`, so no license is needed: only the actors are under test.
"""

import shutil
import time
import typing
import uuid

import pytest
import ray
from ray.actor import ActorHandle
from ray.cluster_utils import Cluster

from polars_nanook_ray.actors import (
    PolarsSchedulerActor,
    list_worker_names,
    resolve_scheduler_name,
    terminate_actors,
)
from polars_nanook_ray.cluster import PolarsRayCluster
from polars_nanook_ray.config import (
    PolarsEnterpriseLicenseConfig,
    PolarsObservatoryConfig,
    PolarsRayClusterConfig,
    PolarsSchedulerConfig,
    PolarsWorkerConfig,
)

NUM_CPU_NODES = 2


@pytest.fixture(scope="module")
def cpu_node_ids() -> typing.Iterator[set[str]]:
    # a head without CPUs, as Ray recommends for multi-node clusters
    cluster = Cluster(
        initialize_head=True,
        head_node_args={
            "num_cpus": 0,
            "resources": {"head": 1},
            "include_dashboard": False,
        },
    )
    nodes = [cluster.add_node(num_cpus=2) for _ in range(NUM_CPU_NODES)]
    cluster.wait_for_nodes()

    ray.init(address=cluster.address)

    yield {node.node_id for node in nodes}

    ray.shutdown()
    cluster.shutdown()


def _config(**kwargs: typing.Any) -> PolarsRayClusterConfig:
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
        worker=PolarsWorkerConfig(cpus_hint=1, memory_hint=1),
        **kwargs,
    )


@pytest.fixture
def config(request: pytest.FixtureRequest) -> PolarsRayClusterConfig:
    return _config(**getattr(request, "param", {}))


@pytest.fixture
def scheduler(
    cpu_node_ids: set[str],  # noqa: ARG001
    config: PolarsRayClusterConfig,
) -> typing.Iterator[ActorHandle]:
    actor = PolarsSchedulerActor.options(  # type: ignore[attr-defined]
        name=resolve_scheduler_name(),
        namespace=config.cluster_id,
        resources={"head": 0.001},
        num_cpus=0,
    ).remote(config)

    yield actor

    PolarsRayCluster(config).stop()
    terminate_actors([actor], config.actor_response_timeout)


def _wait_for_workers(cluster_id: str, expected: int) -> dict[str, ActorHandle]:
    deadline = time.monotonic() + 30
    names = list_worker_names(cluster_id)
    while len(names) != expected and time.monotonic() < deadline:
        time.sleep(0.5)
        names = list_worker_names(cluster_id)

    assert len(names) == expected, sorted(names)
    return {name: ray.get_actor(name, namespace=cluster_id) for name in names}


def _node_id(actor: ActorHandle) -> str:
    return str(
        ray.get(
            actor.__ray_call__.remote(lambda _: ray.get_runtime_context().get_node_id())
        )
    )


def test_one_worker_per_cpu_node(
    cpu_node_ids: set[str],
    config: PolarsRayClusterConfig,
    scheduler: ActorHandle,
) -> None:
    ray.get(scheduler.rescale_worker_pool_to.remote(NUM_CPU_NODES + 1))

    workers = _wait_for_workers(config.cluster_id, NUM_CPU_NODES)

    node_ids = {name: _node_id(actor) for name, actor in workers.items()}
    assert set(node_ids.values()) == cpu_node_ids
    for name, node_id in node_ids.items():
        assert name == f"worker-{node_id[:12]}-0"


def test_repeated_upscale_adds_no_extra_workers(
    config: PolarsRayClusterConfig,
    scheduler: ActorHandle,
) -> None:
    ray.get([scheduler.rescale_worker_pool_to.remote(NUM_CPU_NODES) for _ in range(3)])

    _wait_for_workers(config.cluster_id, NUM_CPU_NODES)


@pytest.mark.parametrize("config", [{"workers_per_node": 2}], indirect=True)
def test_several_workers_per_node_spread_first(
    config: PolarsRayClusterConfig,
    scheduler: ActorHandle,
) -> None:
    ray.get(scheduler.rescale_worker_pool_to.remote(3))

    workers = _wait_for_workers(config.cluster_id, 3)

    assert len({_node_id(actor) for actor in workers.values()}) == NUM_CPU_NODES
    assert sorted(name[-2:] for name in workers) == ["-0", "-0", "-1"]


def test_worker_host_is_the_node_ip(
    config: PolarsRayClusterConfig,
    scheduler: ActorHandle,
) -> None:
    ray.get(scheduler.rescale_worker_pool_to.remote(1))
    (worker,) = _wait_for_workers(config.cluster_id, 1).values()

    node_ips = {node["NodeID"]: node["NodeManagerAddress"] for node in ray.nodes()}
    host = ray.get(worker.__ray_call__.remote(lambda actor: actor.worker_host))

    assert host == node_ips[_node_id(worker)]


def test_stop_without_dashboard(
    config: PolarsRayClusterConfig,
    scheduler: ActorHandle,
) -> None:
    ray.get(scheduler.rescale_worker_pool_to.remote(NUM_CPU_NODES))
    _wait_for_workers(config.cluster_id, NUM_CPU_NODES)

    PolarsRayCluster(config).stop()

    assert list_worker_names(config.cluster_id) == set()


@pytest.mark.parametrize("config", [{"num_workers": NUM_CPU_NODES + 1}], indirect=True)
def test_start_fails_fast_without_room(
    cpu_node_ids: set[str],  # noqa: ARG001
    config: PolarsRayClusterConfig,
) -> None:
    with pytest.raises(RuntimeError, match="Cannot start 3 workers"):
        PolarsRayCluster(config).start()

    assert list_worker_names(config.cluster_id) == set()
