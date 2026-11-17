import functools
import logging
import pathlib
import shutil
import socket
import subprocess
import sys
import sysconfig

import psutil
import toml

logger = logging.getLogger(__name__)

COMPUTE_SCRIPT = "polars-nanook-compute"


def _resolve_compute_script() -> str:
    """Return the path to the console script of the `polars-nanook-compute` package."""
    path = shutil.which(COMPUTE_SCRIPT, path=sysconfig.get_path("scripts")) or (
        shutil.which(COMPUTE_SCRIPT)
    )
    if path is None:
        msg = (
            f"Cannot find the '{COMPUTE_SCRIPT}' script, which the package of the "
            "same name installs on Linux only; set 'binary_path'."
        )
        raise RuntimeError(msg)
    return path


class SubprocessRunner:
    """Run one binary instance for an actor, as a subprocess on the host."""

    def __init__(
        self,
        binary_path: str | None,
        instance_id: str,
        name: str,
    ) -> None:
        self.binary_path = binary_path
        self.instance_id = instance_id
        self.name = name

        self._handle: subprocess.Popen[bytes] | None = None

    def _raise_if_exited(self) -> None:
        """Raise if the binary exited, as nothing restarts it."""
        exit_code = None if self._handle is None else self._handle.poll()

        if exit_code is None:
            return

        reason = (
            f"was killed by signal {-exit_code}"
            if exit_code < 0
            else f"exited with code {exit_code}"
        )
        msg = f"{self.instance_id} {reason}, see the actor logs for its output"
        raise RuntimeError(msg)

    @functools.cached_property
    def command(self) -> list[str]:
        """Return the command line launching the binary, before its arguments."""
        # the script, as workers re-run sys.argv[0] to spawn their executors; through
        # sys.executable, for stop_orphans to match the command line it launched
        return (
            [sys.executable, _resolve_compute_script()]
            if self.binary_path is None
            else [self.binary_path]
        )

    @functools.cached_property
    def host(self) -> str:
        """Return the address the scheduler and workers reach the binary on."""
        return socket.gethostbyname(socket.gethostname())

    def start(self, args: list[str], env: dict[str, str]) -> None:
        """Launch the binary with the arguments given, without waiting for it."""
        self._handle = subprocess.Popen([*self.command, *args], env=env)
        logger.info("Process ID of %s: %d", self.name, self._handle.pid)

    def stop(self) -> None:
        """Stop the binary, killing it on timeout; a no-op when not started."""
        if self._handle is not None:
            logger.info("Stopping %s (%d)", self.name, self._handle.pid)
            self._handle.terminate()
            try:
                self._handle.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._handle.kill()
                self._handle.wait()
        self._handle = None

    def stop_orphans(self, cluster_id: str, ports: list[int]) -> None:
        """Reclaim what a crashed previous instance left, before starting anew."""
        for process in psutil.process_iter():
            try:
                port = next(
                    (
                        conn.laddr.port
                        for conn in process.net_connections(kind="tcp")
                        if conn.status == psutil.CONN_LISTEN
                        and conn.laddr.port in ports
                    ),
                    None,
                )
                if port is None:
                    continue
                cmdline = process.cmdline()
            except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess):
                continue

            msg = f"Port {port} is already in use by process ID {process.pid} "

            if cmdline[: len(self.command)] != self.command:
                msg += "which does not look like one of our own processes."
                raise RuntimeError(msg)

            try:
                stale_config = toml.loads(
                    pathlib.Path(cmdline[len(self.command) + 2]).read_text()
                )
            except (IndexError, OSError) as exc:
                msg += "but we cannot confirm if it is a stale instance."
                raise RuntimeError(msg) from exc

            if stale_config.get("cluster_id") != cluster_id:
                msg += f"by a different cluster ('{stale_config.get('cluster_id')}')."
                raise RuntimeError(msg)

            if stale_config.get("instance_id") != self.instance_id:
                msg += f"by a different instance ('{stale_config.get('instance_id')}')."
                raise RuntimeError(msg)

            logger.warning(
                "Cleaning orphan '%s' process left over from a previous crash",
                self.instance_id,
            )

            process.terminate()
            try:
                process.wait(timeout=10)
            except psutil.TimeoutExpired:
                process.kill()
                process.wait()

    def is_alive(self) -> bool:
        """Return whether the binary is running right now."""
        return self._handle is not None and self._handle.poll() is None

    def is_listening(self, port: int) -> bool:
        """Return whether the binary accepts connections on the port given."""
        try:
            with socket.create_connection((self.host, port), timeout=0.5):
                return True
        except OSError:
            return False

    def is_started(self) -> bool:
        """Return whether the binary was started, and not stopped since."""
        return self._handle is not None

    def pid(self) -> int | None:
        """Return the process ID of the binary on the host, if it is running."""
        return None if self._handle is None else self._handle.pid
