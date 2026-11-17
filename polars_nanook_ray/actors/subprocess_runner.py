import functools
import logging
import pathlib
import re
import secrets
import shutil
import socket
import subprocess
import sys
import sysconfig

import psutil
import toml

logger = logging.getLogger(__name__)


def _service_args(config_path: str | pathlib.Path) -> list[str]:
    """Return the arguments running the scheduler or worker service of a config."""
    return ["service", "--config-path", str(config_path)]


COMPUTE_SCRIPT = "polars-nanook-compute"


def engine_instance_id(name: str) -> str:
    """Return a new engine instance ID for one start of the instance `name`."""
    return f"{name}-{secrets.token_hex(4)}"


def is_engine_instance_of(engine_instance_id: object, name: str) -> bool:
    """Return whether an engine instance ID is `name`, or one start of it."""
    return engine_instance_id == name or (
        isinstance(engine_instance_id, str)
        and re.fullmatch(rf"{re.escape(name)}-[0-9a-f]{{8}}", engine_instance_id)
        is not None
    )


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

    def new_engine_instance_id(self) -> str:
        """Return a new engine instance ID for one start of this instance."""
        return engine_instance_id(self.instance_id)

    def start(self, config_path: pathlib.Path, env: dict[str, str]) -> None:
        """Launch the service of the config given, without waiting for it."""
        args = _service_args(config_path)
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

    def _is_instance_of(self, cmdline: list[str], cluster_id: str) -> bool:
        """Return whether the command line runs this instance of the cluster given."""
        command, args = cmdline[: len(self.command)], cmdline[len(self.command) :]
        if command != self.command or not args or args != _service_args(args[-1]):
            return False

        try:
            config = toml.loads(pathlib.Path(args[-1]).read_text())
        except (OSError, ValueError):  # unreadable, not UTF-8 or not TOML
            return False

        return config.get("cluster_id") == cluster_id and is_engine_instance_of(
            config.get("instance_id"), self.instance_id
        )

    def stop_orphans(self, cluster_id: str) -> None:
        for process in psutil.process_iter():
            try:
                if not self._is_instance_of(process.cmdline(), cluster_id):
                    continue

                logger.warning(
                    "Stopping orphan '%s' process (%d) left over from a previous crash",
                    self.instance_id,
                    process.pid,
                )
                process.terminate()
                try:
                    process.wait(timeout=10)
                except psutil.TimeoutExpired:
                    process.kill()
                    process.wait()
            except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess):
                continue

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
