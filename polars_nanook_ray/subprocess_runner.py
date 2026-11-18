import contextlib
import ctypes
import functools
import logging
import os
import pathlib
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import sysconfig
import tempfile
import threading
import time
import typing

import ray

from polars_nanook_ray.engine_log import EngineLog

if typing.TYPE_CHECKING:
    from polars_nanook_ray.identity import Identity

logger = logging.getLogger(__name__)


COMPUTE_SCRIPT = "polars-nanook-compute"

# for the engine's last lines to reach its log once it exited
DRAIN_TIMEOUT = 2.0
# for an engine that failed to start to stop, as it may not handle SIGTERM yet
FAILED_START_STOP_TIMEOUT = 1.0

_PR_SET_PDEATHSIG = 1


def _die_with_this_process() -> typing.Callable[[], None] | None:
    """Return a `preexec_fn` that has the kernel stop the child when we die, on Linux.

    Our handlers stop the binary when the actor is stopped, but not when the actor's
    process is killed outright, e.g. by the out-of-memory killer. The kernel signals the
    child when the thread that forked it exits; Ray's actor threads live as long as the
    actor. The function only makes system calls, as the child of a threaded process
    must not take locks before it execs.
    """
    if sys.platform != "linux":
        return None

    try:
        prctl = ctypes.CDLL(None, use_errno=True).prctl
    except (AttributeError, OSError):  # e.g. a statically linked Python
        logger.warning(
            "Cannot find prctl: the binary will outlive this process if it is killed"
        )
        return None
    parent = os.getpid()

    def preexec() -> None:
        prctl(_PR_SET_PDEATHSIG, signal.SIGTERM)
        if os.getppid() != parent:  # we died before the signal was set up
            os._exit(1)

    return preexec


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


class EngineStartError(RuntimeError):
    """The engine exited, or did not listen, before it was ever ready."""


def _exit_reason(exit_code: int) -> str:
    return (
        f"was killed by signal {-exit_code}"
        if exit_code < 0
        else f"exited with code {exit_code}"
    )


class SubprocessRunner:
    """Run the binary of the scheduler or a worker, as a subprocess on the host."""

    def __init__(self, binary_path: str | None, identity: "Identity") -> None:
        self.binary_path = binary_path
        self.identity = identity
        self.config_path = pathlib.Path(
            tempfile.gettempdir(),
            f"{identity.actor_name}-{secrets.token_hex(8)}",
            "config.toml",
        )

        self._handle: subprocess.Popen[bytes] | None = None
        self._stopping = threading.Event()
        self._log = EngineLog()
        self._readers: list[threading.Thread] = []
        self._lock = threading.Lock()
        # until the engine listened, its exit is for `wait_until_listening` to raise
        self._raise_exit = False

    def _raise_if_exited(self) -> None:
        """Raise if the binary exited, as nothing restarts it."""
        exit_code = None if self._handle is None else self._handle.poll()

        if exit_code is None:
            return

        self._drain()
        logged = self._logged()
        msg = f"{self.identity} {_exit_reason(exit_code)}" + (
            f": {logged}" if logged else ", see the actor logs for its output"
        )
        raise RuntimeError(msg)

    def _logged(self) -> str:
        """Return the errors, or else warnings, the engine logged, on one line."""
        return "; ".join(self._log.problems())

    def _drain(self) -> None:
        """Wait a little for the output of the exited engine to be read."""
        deadline = time.monotonic() + DRAIN_TIMEOUT
        for reader in self._readers:
            reader.join(timeout=max(deadline - time.monotonic(), 0))

    @functools.cached_property
    def command(self) -> list[str]:
        """Return the command line launching the binary, before its arguments."""
        # the script, as workers re-run sys.argv[0] to spawn their executors
        return (
            [sys.executable, _resolve_compute_script()]
            if self.binary_path is None
            else [self.binary_path]
        )

    @functools.cached_property
    def host(self) -> str:
        """Return the address the scheduler and workers reach the binary on."""
        return ray.util.get_node_ip_address()

    def start(self, config: str, env: typing.Mapping[str, str]) -> None:
        """Write the TOML config given and launch its service, without waiting for it.

        `env` is added to this process's environment. The engine's output still goes
        to this process's stdout and stderr, and its errors are kept to report a
        failed start.
        """
        self._stopping.clear()
        self._raise_exit = True
        self.config_path.parent.mkdir(mode=0o700, exist_ok=True)
        self.config_path.write_text(config)
        self._handle = subprocess.Popen(
            [*self.command, "service", "--config-path", str(self.config_path)],
            env={**os.environ, **env},
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=_die_with_this_process(),
        )
        logger.info("Process ID of %s: %d", self.identity, self._handle.pid)
        self._log = EngineLog()
        self._readers = [
            threading.Thread(
                target=self._forward,
                args=(source, target, self._log),
                name=f"{self.identity}-{target}",
                daemon=True,
            )
            for source, target in (
                (self._handle.stdout, "stdout"),
                (self._handle.stderr, "stderr"),
            )
        ]
        for reader in self._readers:
            reader.start()
        threading.Thread(
            target=self._exit_with_binary,
            args=(self._handle,),
            name=f"{self.identity}-watch",
            daemon=True,
        ).start()

    @staticmethod
    def _forward(source: typing.IO[bytes] | None, target: str, log: EngineLog) -> None:
        """Copy each line of `source` to `sys.<target>`, and keep it in `log`."""
        if source is None:
            return
        with source:
            for line in iter(source.readline, b""):
                stream = getattr(sys, target)
                # the engine blocks on a full pipe: keep reading whatever happens here
                with contextlib.suppress(ValueError, OSError):
                    if (buffer := getattr(stream, "buffer", None)) is not None:
                        buffer.write(line)
                        buffer.flush()
                    else:
                        stream.write(line.decode(errors="replace"))
                        stream.flush()
                log.add(line.decode(errors="replace"))

    def wait_until_listening(self, port: int, timeout: float) -> None:
        """Wait until the engine accepts connections on `port`.

        Raises `EngineStartError`, with the errors the engine logged, if it exits
        first or does not listen within `timeout` seconds; it then stays running.
        """
        handle = self._handle
        if handle is None:
            msg = f"{self.identity} was not started"
            raise RuntimeError(msg)

        deadline = time.monotonic() + timeout
        try:
            while handle.poll() is None and not self.is_listening(port):
                if time.monotonic() >= deadline:
                    logged = self._logged()
                    msg = (
                        f"Did not start within {timeout:g}s: {self.identity}, whose "
                        f"engine does not listen on port {port}."
                    ) + (f" Its engine logged: {logged}" if logged else "")
                    raise EngineStartError(msg)
                time.sleep(0.2)
        finally:
            with self._lock:
                exit_code = handle.poll()
                self._raise_exit = exit_code is not None

        if exit_code is not None:
            self._drain()
            logged = self._logged()
            msg = f"The {self.identity} engine {_exit_reason(exit_code)} while starting"
            msg += f": {logged}" if logged else ", logging no error"
            raise EngineStartError(msg)

    def _exit_with_binary(self, handle: "subprocess.Popen[bytes]") -> None:
        """Exit the actor process when the binary exits unasked, for Ray to restart it.

        Ray only sees the actor: without this, an actor whose binary died stays alive
        and its restart limit never applies. An exit while starting is raised by
        `wait_until_listening` instead, as restarting does not fix it.
        """
        exit_code = handle.wait()
        with self._lock:
            if self._stopping.is_set() or self._raise_exit:
                return
        self._drain()
        logger.error(
            "%s (%d) exited with code %d; exiting its actor",
            self.identity,
            handle.pid,
            exit_code,
        )
        if logged := self._logged():
            logger.error("%s logged: %s", self.identity, logged)
        logging.shutdown()
        os._exit(1)

    def stop(self, timeout: float = 10.0) -> None:
        """Stop the binary, killing it after `timeout` seconds; remove its config."""
        self._stopping.set()
        if self._handle is not None:
            logger.info("Stopping %s (%d)", self.identity, self._handle.pid)
            self._handle.terminate()
            try:
                self._handle.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                self._handle.kill()
                self._handle.wait()
        self._handle = None
        shutil.rmtree(self.config_path.parent, ignore_errors=True)

    def is_ready(self, port: int) -> bool:
        """Return whether the binary runs and accepts connections on `port`.

        Raises if it exited, as nothing restarts it.
        """
        self._raise_if_exited()
        return self.is_alive() and self.is_listening(port)

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
