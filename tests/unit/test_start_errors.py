import pathlib
import socket
import sys
import time
import typing

import pytest
import ray
from ray.cluster_utils import Cluster

from polars_nanook_ray import EngineStartError, Ports, RayClusterContext
from polars_nanook_ray.actors import WorkerExited

LICENSE_ERROR = "License file check failed: Failed to read license file"

# Stands in for the engine: logs a fatal error as the engine does, then exits or
# hangs; or listens on its scheduler's or worker's port, as a healthy engine does.
_FAKE_ENGINE = """\
#!{python}
import json, pathlib, signal, socket, sys, time, tomllib

with open("{runs}", "a") as runs:
    runs.write("run\\n")
config = tomllib.loads(pathlib.Path(sys.argv[3]).read_text())
role = "scheduler" if "scheduler" in config else "worker"
behaviour = {behaviours}[role]
if behaviour == "listen":
    if role == "scheduler":
        addr = config["scheduler"]["worker_service"]["bind_addr"]
    else:
        addr = config["worker"]["task_service"]["bind_addr"]
    server = socket.socket()
    server.bind(("", int(addr.rsplit(":", 1)[1])))
    server.listen()
    time.sleep(600)
error = {{"message": "{error}\\n\\nStack backtrace:\\n   0: <unknown>", "error": "{error}"}}
print(json.dumps({{"fields": error, "level": "ERROR"}}), flush=True)
print('"{error}"', file=sys.stderr, flush=True)
if behaviour == "hang":
    signal.signal(signal.SIGTERM, signal.SIG_IGN)  # as an engine stuck starting
    time.sleep(600)
sys.exit(1)
"""


@pytest.fixture(scope="module")
def ray_cluster() -> typing.Iterator[None]:
    cluster = Cluster(initialize_head=True, head_node_args={"num_cpus": 4})
    ray.init(address=cluster.address)
    yield
    ray.shutdown()
    cluster.shutdown()


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("", 0))
        return int(sock.getsockname()[1])


class _Engine(typing.NamedTuple):
    path: str
    runs: pathlib.Path

    def run_count(self) -> int:
        return len(self.runs.read_text().splitlines()) if self.runs.exists() else 0


def _engine(tmp_path: pathlib.Path, scheduler: str, worker: str = "listen") -> _Engine:
    runs = tmp_path / "runs"
    binary = tmp_path / "engine"
    binary.write_text(
        _FAKE_ENGINE.format(
            python=sys.executable,
            runs=runs,
            behaviours={"scheduler": scheduler, "worker": worker},
            error=LICENSE_ERROR,
        )
    )
    binary.chmod(0o755)
    return _Engine(str(binary), runs)


def _context(
    tmp_path: pathlib.Path, engine: _Engine, workers: int
) -> RayClusterContext:
    return RayClusterContext(
        license="/dev/null",
        name=f"start-errors-{tmp_path.name}",
        binary_path=engine.path,
        workers=workers,
        cpus=0,
        memory=0,
        startup_timeout=3,
        observatory=False,
        ports=Ports(client=None, dashboard=None),
    )


def _actors(ctx: RayClusterContext) -> list[str]:
    return [
        actor["name"]
        for actor in ray.util.list_named_actors(all_namespaces=True)
        if actor["namespace"] == ctx._spec.name
    ]


@pytest.mark.usefixtures("ray_cluster")
def test_scheduler_engine_exiting_shows_its_error_and_is_not_retried(
    tmp_path: pathlib.Path,
) -> None:
    engine = _engine(tmp_path, scheduler="exit")
    ctx = _context(tmp_path, engine, workers=0)

    with pytest.raises(EngineStartError) as raised:
        ctx.start()

    assert str(raised.value) == (
        f"The scheduler engine exited with code 1 while starting: {LICENSE_ERROR}"
    )
    assert engine.run_count() == 1
    assert _actors(ctx) == []


@pytest.mark.usefixtures("ray_cluster")
def test_worker_engine_exiting_shows_its_error_and_stops_the_cluster(
    tmp_path: pathlib.Path,
) -> None:
    engine = _engine(tmp_path, scheduler="listen", worker="exit")
    ctx = _context(tmp_path, engine, workers=1)

    with pytest.raises(WorkerExited) as raised:
        ctx.start()

    assert str(raised.value).startswith("The worker-")
    assert str(raised.value).endswith(
        f" engine exited with code 1 while starting: {LICENSE_ERROR}"
    )
    assert engine.run_count() == 2  # the scheduler, and the worker once
    assert _actors(ctx) == []


@pytest.mark.usefixtures("ray_cluster")
def test_scheduler_engine_never_listening_times_out_with_its_logged_error(
    tmp_path: pathlib.Path,
) -> None:
    engine = _engine(tmp_path, scheduler="hang")
    ctx = _context(tmp_path, engine, workers=0)

    started = time.monotonic()
    with pytest.raises(EngineStartError) as raised:
        ctx.start()

    assert time.monotonic() - started < 3 * ctx._spec.startup_timeout
    assert str(raised.value).startswith("Did not start within 3s: scheduler")
    assert str(raised.value).endswith(f"Its engine logged: {LICENSE_ERROR}")
    assert _actors(ctx) == []
