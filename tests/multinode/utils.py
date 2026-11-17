import pathlib
import socket
import sys
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

# an engine that fails once it has read its config, as on a bad license
FAILING_ENGINE = pathlib.Path(tempfile.gettempdir()) / "polars-nanook-ray-stub-fail"
FAILING_ENGINE.write_text("#!/bin/sh\nsleep 1\nexit 1\n")
FAILING_ENGINE.chmod(0o755)

# an engine that listens on the ports of its config, for the actors to be ready
LISTENING_ENGINE = pathlib.Path(tempfile.gettempdir()) / "polars-nanook-ray-stub-listen"
LISTENING_ENGINE.write_text(f"""#!{sys.executable}
import socket, sys, time, tomllib

with open(sys.argv[sys.argv.index("--config-path") + 1], "rb") as file:
    config = tomllib.load(file)

def bind_addrs(node):
    for key, value in node.items():
        if isinstance(value, dict):
            yield from bind_addrs(value)
        elif key == "bind_addr":
            yield value

listeners = []
for addr in bind_addrs(config):
    listener = socket.socket()
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        listener.bind(("", int(addr.rsplit(":", 1)[1])))
    except OSError:  # a fixed port shared by every engine on this one machine
        continue
    listener.listen()
    listeners.append(listener)
time.sleep(3600)
""")
LISTENING_ENGINE.chmod(0o755)


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("", 0))
        return int(sock.getsockname()[1])


def make_config(**kwargs: typing.Any) -> PolarsRayClusterConfig:
    # an engine that keeps running, or its actor exits with it
    kwargs.setdefault("binary_path", str(_STUB_ENGINE))
    kwargs.setdefault(
        "scheduler",
        PolarsSchedulerConfig(
            cpus_hint=0,
            memory_hint=1,
            client_port=free_port(),
            worker_registration_port=free_port(),
            observatory=PolarsObservatoryConfig(enabled=False),
        ),
    )
    kwargs.setdefault("worker", PolarsWorkerConfig(cpus_hint=1, memory_hint=1))
    return PolarsRayClusterConfig(
        cluster_id=f"polars-nanook-{uuid.uuid4().hex[:8]}",
        license=PolarsEnterpriseLicenseConfig(license_path="/nonexistent.json"),
        **kwargs,
    )


def run_scheduler(
    config: PolarsRayClusterConfig, max_restarts: int = 0
) -> typing.Iterator[ActorHandle]:
    """Yield a scheduler actor on the head node, then stop it and its workers."""
    actor = PolarsSchedulerActor.options(  # type: ignore[attr-defined]
        name=SCHEDULER.actor_name,
        namespace=config.cluster_id,
        max_restarts=max_restarts,
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
