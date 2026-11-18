import json
import threading
import time
import typing
import urllib.error
import urllib.request

import pytest

from polars_nanook_ray._spec import ClusterSpec
from polars_nanook_ray.actors.pool import WorkerPool
from polars_nanook_ray.actors.scaling import serve_scaling
from polars_nanook_ray.identity import WorkerId

if typing.TYPE_CHECKING:
    from polars_nanook_ray.actors.scaling import ScalingEndpoint

from .test_pool import InMemoryWorkers, config


class Response(typing.NamedTuple):
    status: int
    body: typing.Any


@pytest.fixture
def workers() -> InMemoryWorkers:
    return InMemoryWorkers()


Serve = typing.Callable[[WorkerPool, ClusterSpec], int]


@pytest.fixture
def serve() -> typing.Iterator[Serve]:
    """Serve scaling requests; return the port served on."""
    endpoints: list[ScalingEndpoint] = []

    def _serve(worker_pool: WorkerPool, cluster: ClusterSpec) -> int:
        endpoints.append(serve_scaling(worker_pool, cluster))
        return endpoints[-1].port

    yield _serve

    for endpoint in endpoints:
        endpoint.close()


def scaling_config(**bounds: typing.Any) -> ClusterSpec:
    return config(autoscale=bounds)


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
    workers: InMemoryWorkers, serve: Serve
) -> None:
    cluster = scaling_config(min_workers=1, max_workers=8)
    worker_pool = WorkerPool(workers, cluster, 0)
    worker_pool.ensure(2)
    port = serve(worker_pool, cluster)

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
def cluster() -> ClusterSpec:
    return scaling_config()


def test_scale_config_reports_an_unbounded_max_as_null(
    workers: InMemoryWorkers,
    cluster: ClusterSpec,
    serve: Serve,
) -> None:
    port = serve(WorkerPool(workers, cluster, 0), cluster)

    assert request(port, "GET", "/scale_config") == (
        200,
        {"available": 0, "desired": 0, "min": 0, "max": None},
    )


def test_scale_to_starts_workers(
    workers: InMemoryWorkers,
    cluster: ClusterSpec,
    serve: Serve,
) -> None:
    port = serve(WorkerPool(workers, cluster, 0), cluster)

    assert request(port, "POST", "/scale_to", {"amount": 2}) == (204, None)
    assert workers.running() == {WorkerId(0), WorkerId(1)}
    assert request(port, "GET", "/scale_config").body["desired"] == 2


def test_scale_to_is_clamped_to_the_bounds(
    workers: InMemoryWorkers, serve: Serve
) -> None:
    cluster = scaling_config(max_workers=1)
    port = serve(WorkerPool(workers, cluster, 0), cluster)

    assert request(port, "POST", "/scale_to", {"amount": 3}).status == 204
    assert workers.running() == {WorkerId(0)}


def test_scale_to_deletes_the_workers_named(
    workers: InMemoryWorkers,
    cluster: ClusterSpec,
    serve: Serve,
) -> None:
    worker_pool = WorkerPool(workers, cluster, 0)
    worker_pool.ensure(3)
    port = serve(worker_pool, cluster)

    body = {"amount": 2, "workers_to_delete": [WorkerId(0).actor_name]}

    assert request(port, "POST", "/scale_to", body).status == 204
    assert workers.running() == {WorkerId(1), WorkerId(2)}


def test_scale_to_keeps_the_workers_named(
    workers: InMemoryWorkers,
    cluster: ClusterSpec,
    serve: Serve,
) -> None:
    worker_pool = WorkerPool(workers, cluster, 0)
    worker_pool.ensure(3)
    port = serve(worker_pool, cluster)

    body = {"amount": 1, "workers_to_keep": [WorkerId(2).actor_name]}

    assert request(port, "POST", "/scale_to", body).status == 204
    assert workers.running() == {WorkerId(2)}


@pytest.mark.parametrize(
    "body",
    [{"amount": -1}, {"amount": "2"}, {"amount": 1, "workers_to_keep": [1]}, {}],
)
def test_invalid_scale_to_is_rejected(
    workers: InMemoryWorkers,
    cluster: ClusterSpec,
    serve: Serve,
    body: dict,
) -> None:
    port = serve(WorkerPool(workers, cluster, 0), cluster)

    response = request(port, "POST", "/scale_to", body)

    assert response.status == 400
    assert response.body["detail"]
    assert workers.running() == set()


def test_non_json_scale_to_is_unsupported(
    workers: InMemoryWorkers,
    cluster: ClusterSpec,
    serve: Serve,
) -> None:
    port = serve(WorkerPool(workers, cluster, 0), cluster)

    response = request(port, "POST", "/scale_to", {"amount": 1}, "text/plain")

    assert response.status == 415


@pytest.mark.parametrize(("method", "path"), [("GET", "/scale_to"), ("POST", "/nope")])
def test_other_paths_are_not_found(
    workers: InMemoryWorkers,
    cluster: ClusterSpec,
    serve: Serve,
    method: str,
    path: str,
) -> None:
    port = serve(WorkerPool(workers, cluster, 0), cluster)

    assert request(port, method, path, {"amount": 1}) == (404, {"detail": "not found"})


def test_a_failed_rescale_is_a_server_error(
    cluster: ClusterSpec,
    serve: Serve,
) -> None:
    workers = FailingWorkers()
    worker_pool = WorkerPool(workers, cluster, 0)
    worker_pool.ensure(1)
    port = serve(worker_pool, cluster)

    assert request(port, "POST", "/scale_to", {"amount": 0}) == (
        500,
        {"detail": "the GCS is unreachable"},
    )


def test_scale_config_is_answered_during_a_slow_scale_down(
    cluster: ClusterSpec,
    serve: Serve,
) -> None:
    workers = StuckWorkers()
    worker_pool = WorkerPool(workers, cluster, 0)
    worker_pool.ensure(2)
    port = serve(worker_pool, cluster)

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


def test_endpoints_serve_on_distinct_free_ports(
    workers: InMemoryWorkers, cluster: ClusterSpec, serve: Serve
) -> None:
    worker_pool = WorkerPool(workers, cluster, 0)

    ports = {serve(worker_pool, cluster), serve(worker_pool, cluster)}

    assert len(ports) == 2
    for port in ports:
        assert request(port, "GET", "/scale_config").status == 200


def test_a_closed_endpoint_stops_answering(
    workers: InMemoryWorkers, cluster: ClusterSpec
) -> None:
    endpoint = serve_scaling(WorkerPool(workers, cluster, 0), cluster)
    assert request(endpoint.port, "GET", "/scale_config").status == 200

    endpoint.close()

    with pytest.raises(urllib.error.URLError):
        request(endpoint.port, "GET", "/scale_config")


def test_close_twice_and_during_a_slow_request(cluster: ClusterSpec) -> None:
    workers = StuckWorkers()
    worker_pool = WorkerPool(workers, cluster, 0)
    worker_pool.ensure(1)
    endpoint = serve_scaling(worker_pool, cluster)

    scale_down = threading.Thread(
        target=request, args=(endpoint.port, "POST", "/scale_to", {"amount": 0})
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
