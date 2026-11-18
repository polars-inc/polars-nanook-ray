import json
import socket
import threading
import time
import typing
import urllib.error
import urllib.request

import pytest

from polars_nanook_ray.actors.pool import WorkerPool
from polars_nanook_ray.actors.scaling import ScalingEndpoint, serve_scaling
from polars_nanook_ray.config import (
    PolarsRayClusterConfig,
    PolarsScalingConfig,
    PolarsSchedulerConfig,
)
from polars_nanook_ray.identity import WorkerId
from polars_nanook_ray.ports import PortInUseError, resolve_ports

from .test_pool import InMemoryWorkers, config


class Response(typing.NamedTuple):
    status: int
    body: typing.Any


@pytest.fixture
def workers() -> InMemoryWorkers:
    return InMemoryWorkers()


@pytest.fixture
def port() -> int:
    return resolve_ports({"scaling.port": None})["scaling.port"]


@pytest.fixture
def serve() -> typing.Iterator[typing.Callable[..., ScalingEndpoint]]:
    endpoints: list[ScalingEndpoint] = []

    def _serve(
        worker_pool: WorkerPool, cluster: PolarsRayClusterConfig
    ) -> ScalingEndpoint:
        endpoints.append(serve_scaling(worker_pool, cluster))
        return endpoints[-1]

    yield _serve

    for endpoint in endpoints:
        endpoint.close()


def scaling_config(port: int, **kwargs: typing.Any) -> PolarsRayClusterConfig:
    return config(
        scheduler=PolarsSchedulerConfig(
            scaling=PolarsScalingConfig(enabled=True, port=port)
        ),
        **kwargs,
    )


def request(
    port: int,
    method: str,
    path: str,
    body: object = None,
    content_type: str = "application/json",
) -> Response:
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=data,
        method=method,
        headers={"Content-Type": content_type},
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            raw = response.read()
            return Response(response.status, json.loads(raw) if raw else None)
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        return Response(exc.code, json.loads(raw) if raw else None)


def test_scale_config_reports_the_pool_and_its_bounds(
    workers: InMemoryWorkers, port: int, serve: typing.Callable[..., ScalingEndpoint]
) -> None:
    cluster = scaling_config(port, min_workers=1, max_workers=8)
    worker_pool = WorkerPool(workers, cluster, 0)
    worker_pool.ensure(2)
    serve(worker_pool, cluster)

    assert request(port, "GET", "/scale_config") == (
        200,
        {"available": 2, "desired": 2, "min": 1, "max": 8},
    )


class StuckWorkers(InMemoryWorkers):
    """Workers whose removal waits until released, as a slow scale-down does."""

    def __init__(self) -> None:
        super().__init__()
        self.terminating = threading.Event()
        self.release = threading.Event()

    def terminate(self, workers: typing.Collection[WorkerId]) -> None:
        self.terminating.set()
        self.release.wait(10)
        super().terminate(workers)


class FailingWorkers(InMemoryWorkers):
    def terminate(self, workers: typing.Collection[WorkerId]) -> None:  # noqa: ARG002
        msg = "the GCS is unreachable"
        raise RuntimeError(msg)


@pytest.fixture
def cluster(port: int) -> PolarsRayClusterConfig:
    return scaling_config(port)


def test_scale_config_reports_an_unbounded_max_as_null(
    workers: InMemoryWorkers,
    cluster: PolarsRayClusterConfig,
    port: int,
    serve: typing.Callable[..., ScalingEndpoint],
) -> None:
    serve(WorkerPool(workers, cluster, 0), cluster)

    assert request(port, "GET", "/scale_config") == (
        200,
        {"available": 0, "desired": 0, "min": 0, "max": None},
    )


def test_scale_to_starts_workers(
    workers: InMemoryWorkers,
    cluster: PolarsRayClusterConfig,
    port: int,
    serve: typing.Callable[..., ScalingEndpoint],
) -> None:
    serve(WorkerPool(workers, cluster, 0), cluster)

    assert request(port, "POST", "/scale_to", {"amount": 2}) == (204, None)
    assert workers.running() == {WorkerId(0), WorkerId(1)}
    assert request(port, "GET", "/scale_config").body["desired"] == 2


def test_scale_to_is_clamped_to_the_bounds(
    workers: InMemoryWorkers, port: int, serve: typing.Callable[..., ScalingEndpoint]
) -> None:
    cluster = scaling_config(port, max_workers=1)
    serve(WorkerPool(workers, cluster, 0), cluster)

    assert request(port, "POST", "/scale_to", {"amount": 3}).status == 204
    assert workers.running() == {WorkerId(0)}


def test_scale_to_deletes_the_workers_named(
    workers: InMemoryWorkers,
    cluster: PolarsRayClusterConfig,
    port: int,
    serve: typing.Callable[..., ScalingEndpoint],
) -> None:
    worker_pool = WorkerPool(workers, cluster, 0)
    worker_pool.ensure(3)
    serve(worker_pool, cluster)

    body = {"amount": 2, "workers_to_delete": [WorkerId(0).actor_name]}

    assert request(port, "POST", "/scale_to", body).status == 204
    assert workers.running() == {WorkerId(1), WorkerId(2)}


def test_scale_to_keeps_the_workers_named(
    workers: InMemoryWorkers,
    cluster: PolarsRayClusterConfig,
    port: int,
    serve: typing.Callable[..., ScalingEndpoint],
) -> None:
    worker_pool = WorkerPool(workers, cluster, 0)
    worker_pool.ensure(3)
    serve(worker_pool, cluster)

    body = {"amount": 1, "workers_to_keep": [WorkerId(2).actor_name]}

    assert request(port, "POST", "/scale_to", body).status == 204
    assert workers.running() == {WorkerId(2)}


@pytest.mark.parametrize(
    "body",
    [{"amount": -1}, {"amount": "2"}, {"amount": 1, "workers_to_keep": [1]}, {}],
)
def test_invalid_scale_to_is_rejected(
    workers: InMemoryWorkers,
    cluster: PolarsRayClusterConfig,
    port: int,
    serve: typing.Callable[..., ScalingEndpoint],
    body: dict,
) -> None:
    serve(WorkerPool(workers, cluster, 0), cluster)

    response = request(port, "POST", "/scale_to", body)

    assert response.status == 400
    assert response.body["detail"]
    assert workers.running() == set()


def test_non_json_scale_to_is_unsupported(
    workers: InMemoryWorkers,
    cluster: PolarsRayClusterConfig,
    port: int,
    serve: typing.Callable[..., ScalingEndpoint],
) -> None:
    serve(WorkerPool(workers, cluster, 0), cluster)

    response = request(port, "POST", "/scale_to", {"amount": 1}, "text/plain")

    assert response.status == 415


@pytest.mark.parametrize(("method", "path"), [("GET", "/scale_to"), ("POST", "/nope")])
def test_other_paths_are_not_found(
    workers: InMemoryWorkers,
    cluster: PolarsRayClusterConfig,
    port: int,
    serve: typing.Callable[..., ScalingEndpoint],
    method: str,
    path: str,
) -> None:
    serve(WorkerPool(workers, cluster, 0), cluster)

    assert request(port, method, path, {"amount": 1}) == (404, {"detail": "not found"})


def test_a_failed_rescale_is_a_server_error(
    port: int,
    cluster: PolarsRayClusterConfig,
    serve: typing.Callable[..., ScalingEndpoint],
) -> None:
    workers = FailingWorkers()
    worker_pool = WorkerPool(workers, cluster, 0)
    worker_pool.ensure(1)
    serve(worker_pool, cluster)

    assert request(port, "POST", "/scale_to", {"amount": 0}) == (
        500,
        {"detail": "the GCS is unreachable"},
    )


def test_scale_config_is_answered_during_a_slow_scale_down(
    port: int,
    cluster: PolarsRayClusterConfig,
    serve: typing.Callable[..., ScalingEndpoint],
) -> None:
    workers = StuckWorkers()
    worker_pool = WorkerPool(workers, cluster, 0)
    worker_pool.ensure(2)
    serve(worker_pool, cluster)

    scale_down = threading.Thread(
        target=request, args=(port, "POST", "/scale_to", {"amount": 0})
    )
    scale_down.start()
    try:
        assert workers.terminating.wait(5)

        assert request(port, "GET", "/scale_config").body == {
            "available": 2,
            "desired": 0,
            "min": 0,
            "max": None,
        }
    finally:
        workers.release.set()
        scale_down.join(10)


def test_a_taken_port_is_reported(
    workers: InMemoryWorkers, cluster: PolarsRayClusterConfig, port: int
) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", port))
        listener.listen()

        with pytest.raises(PortInUseError, match=rf"scaling.port {port} .*in use"):
            serve_scaling(WorkerPool(workers, cluster, 0), cluster)


def test_a_closed_endpoint_frees_its_port_for_the_next_one(
    workers: InMemoryWorkers, cluster: PolarsRayClusterConfig, port: int
) -> None:
    worker_pool = WorkerPool(workers, cluster, 0)
    endpoint = serve_scaling(worker_pool, cluster)
    # the endpoint closes the connection first, which lingers in TIME_WAIT on its port
    with socket.create_connection(("127.0.0.1", port), timeout=10) as conn:
        conn.sendall(b"GET /scale_config HTTP/1.0\r\n\r\n")
        while conn.recv(4096):
            pass
    endpoint.close()

    with pytest.raises(urllib.error.URLError):
        request(port, "GET", "/scale_config")

    endpoint = serve_scaling(worker_pool, cluster)
    try:
        assert request(port, "GET", "/scale_config").status == 200
    finally:
        endpoint.close()


def test_close_twice_and_during_a_slow_request(
    port: int, cluster: PolarsRayClusterConfig
) -> None:
    workers = StuckWorkers()
    worker_pool = WorkerPool(workers, cluster, 0)
    worker_pool.ensure(1)
    endpoint = serve_scaling(worker_pool, cluster)

    scale_down = threading.Thread(
        target=request, args=(port, "POST", "/scale_to", {"amount": 0})
    )
    scale_down.start()
    try:
        assert workers.terminating.wait(5)

        started = time.monotonic()
        endpoint.close()
        endpoint.close()

        assert time.monotonic() - started < 2
    finally:
        workers.release.set()
        scale_down.join(10)
