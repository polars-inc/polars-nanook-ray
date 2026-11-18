import contextlib
import errno
import socket
import typing


class PortInUseError(RuntimeError):
    """A fixed port is already bound on this node."""

    @classmethod
    def of(cls, name: str, port: int | None) -> "PortInUseError":
        return cls(f"{name} {port} is already in use on {socket.gethostname()}")


def resolve_ports(requested: typing.Mapping[str, int | None]) -> dict[str, int]:
    """Return the port to bind for each name, picking a free one on this node if unset.

    Raise `PortInUseError` if a fixed port is taken. The ports are released on return,
    so another process may take a picked port before the engine binds it.
    """
    with contextlib.ExitStack() as stack:
        ports: dict[str, int] = {}
        # fixed first, so a picked port never takes one requested later
        for name, port in sorted(requested.items(), key=lambda item: item[1] is None):
            sock = stack.enter_context(
                socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            )
            # as the engine binds, so connections lingering in TIME_WAIT don't count
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("", port or 0))
            except OSError as exc:
                if exc.errno != errno.EADDRINUSE:
                    raise
                raise PortInUseError.of(name, port) from exc
            # Linux may share a port between unlistened SO_REUSEADDR sockets
            sock.listen()
            ports[name] = sock.getsockname()[1]
        return {name: ports[name] for name in requested}
