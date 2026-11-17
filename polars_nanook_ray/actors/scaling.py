import errno
import http.server
import json
import logging
import threading

from pydantic import BaseModel, ConfigDict, Field, StrictStr, ValidationError

from polars_nanook_ray.actors.pool import WorkerPool
from polars_nanook_ray.config import SCALER_HOST, PolarsRayClusterConfig
from polars_nanook_ray.ports import PortInUseError

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

        self._respond(
            200,
            {
                "available": len(status.running),
                "desired": status.target,
                "min": self.server.config.min_workers,
                "max": self.server.config.max_workers,
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
    """A `ThreadingHTTPServer` that carries the worker pool to its handlers.

    Rebinds a port whose closed connections linger in TIME_WAIT, as a restarted
    scheduler must, but not one another process listens on.
    """

    allow_reuse_address = True

    def __init__(self, pool: WorkerPool, config: PolarsRayClusterConfig) -> None:
        self.pool = pool
        self.config = config
        super().__init__(
            (SCALER_HOST, config.scheduler.scaling.port), _ScalingRequestHandler
        )


class ScalingEndpoint:
    """The HTTP endpoint the engine rescales the worker pool through, on 127.0.0.1."""

    def __init__(self, server: _ScalingHTTPServer) -> None:
        self._server: _ScalingHTTPServer | None = server
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


def serve_scaling(pool: WorkerPool, config: PolarsRayClusterConfig) -> ScalingEndpoint:
    """Serve the engine's scaling requests on `scaling.port` until closed.

    Listens on return. Raises `PortInUseError` if the port is taken.
    """
    name, port = "scaling.port", config.scheduler.scaling.port
    try:
        server = _ScalingHTTPServer(pool, config)
    except OSError as exc:
        if exc.errno != errno.EADDRINUSE:
            raise
        raise PortInUseError.of(name, port) from exc

    logger.info("Scaling endpoint listening on %s:%d", SCALER_HOST, port)
    return ScalingEndpoint(server)
