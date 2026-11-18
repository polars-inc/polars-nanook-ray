import http.server
import json
import logging
import threading
import typing

from pydantic import BaseModel, ConfigDict, Field, StrictStr, ValidationError

from polars_nanook_ray._engine import SCALER_HOST
from polars_nanook_ray.actors.pool import WorkerPool
from polars_nanook_ray.options import Autoscale

if typing.TYPE_CHECKING:
    from polars_nanook_ray._spec import ClusterSpec

logger = logging.getLogger(__name__)


class ScaleToRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    amount: int = Field(ge=0, strict=True)
    workers_to_keep: set[StrictStr] | None = None
    workers_to_delete: set[StrictStr] | None = None


class _ScalingRequestHandler(http.server.BaseHTTPRequestHandler):
    """Answer the engine's scaling calls from the worker pool."""

    server: "_ScalingHTTPServer"

    def _respond(self, status: int, payload: dict | None = None) -> None:
        body = json.dumps(payload).encode() if payload is not None else b""

        self.send_response(status)

        if body:
            self.send_header("Content-Type", "application/json")

        self.send_header("Content-Length", str(len(body)))
        self.end_headers()

        if body:
            self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path != "/scale_config":
            self._respond(404, {"detail": "not found"})
            return

        try:
            status = self.server.pool.status()
        except Exception as exc:
            logger.exception("scale_config request failed")
            self._respond(500, {"detail": str(exc)})
            return

        bounds = self.server.spec.autoscale or Autoscale()
        self._respond(
            200,
            {
                "available": len(status.running),
                "desired": status.target,
                "min": bounds.min_workers,
                "max": bounds.max_workers,
            },
        )

    def do_POST(self) -> None:
        if self.path != "/scale_to":
            self._respond(404, {"detail": "not found"})
            return

        if self.headers.get_content_type() != "application/json":
            self._respond(415, {"detail": "expected an application/json request body"})
            return

        length = int(self.headers.get("Content-Length", 0))

        try:
            request = ScaleToRequest.model_validate_json(self.rfile.read(length))
        except ValidationError as exc:
            logger.exception("scale_to request rejected")
            self._respond(
                400,
                {
                    "detail": exc.errors(
                        include_url=False,
                        include_input=False,
                        include_context=False,
                    )
                },
            )
            return

        logger.info(
            "scale_to request: amount=%d, keep=%s, delete=%s",
            request.amount,
            request.workers_to_keep,
            request.workers_to_delete,
        )

        try:
            self.server.pool.rescale(
                request.amount,
                keep=request.workers_to_keep,
                delete=request.workers_to_delete,
            )
        except Exception as exc:
            logger.exception("scale_to request failed")
            self._respond(500, {"detail": str(exc)})
            return

        self._respond(204)

    def log_message(self, format: str, *args: object) -> None:
        logger.debug(format, *args)


class _ScalingHTTPServer(http.server.ThreadingHTTPServer):
    """A `ThreadingHTTPServer` that carries the worker pool to its handlers."""

    def __init__(self, pool: WorkerPool, spec: "ClusterSpec") -> None:
        self.pool = pool
        self.spec = spec
        super().__init__((SCALER_HOST, 0), _ScalingRequestHandler)


class ScalingEndpoint:
    """The HTTP endpoint the engine rescales the worker pool through, on 127.0.0.1."""

    def __init__(self, server: _ScalingHTTPServer) -> None:
        self._server: _ScalingHTTPServer | None = server
        self.port: int = server.server_address[1]
        threading.Thread(
            target=server.serve_forever, name="scaling-endpoint", daemon=True
        ).start()

    def close(self) -> None:
        """Stop serving and release the port; leave requests in progress to finish.

        Safe to call again, also from a signal handler interrupting a call.
        """
        server, self._server = self._server, None
        if server is not None:
            server.shutdown()
            server.server_close()


def serve_scaling(pool: WorkerPool, spec: "ClusterSpec") -> ScalingEndpoint:
    """Serve the engine's scaling requests on a free port until closed.

    Listens on return, on the endpoint's `port`.
    """
    endpoint = ScalingEndpoint(_ScalingHTTPServer(pool, spec))
    logger.info("Scaling endpoint listening on %s:%d", SCALER_HOST, endpoint.port)
    return endpoint
