import ctypes
import functools
import logging
import os
import pathlib
import shutil
import signal
import socket
import subprocess
import sys
import sysconfig
import threading
import typing

import ray

if typing.TYPE_CHECKING:
    from polars_nanook_ray.identity import Identity

logger = logging.getLogger(__name__)


def _service_args(config_path: str | pathlib.Path) -> list[str]:
    """Return the arguments running the scheduler or worker service of a config."""
    return ["service", "--config-path", str(config_path)]


COMPUTE_SCRIPT = "polars-nanook-compute"

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

    prctl = ctypes.CDLL(None, use_errno=True).prctl
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


class SubprocessRunner:
    """Run the binary of the scheduler or a worker, as a subprocess on the host."""

    def __init__(self, binary_path: str | None, identity: "Identity") -> None:
        self.binary_path = binary_path
        self.identity = identity

        self._handle: subprocess.Popen[bytes] | None = None
        self._stopping = threading.Event()

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
        msg = f"{self.identity} {reason}, see the actor logs for its output"
        raise RuntimeError(msg)

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

    def start(self, config_path: pathlib.Path, env: dict[str, str]) -> None:
        """Launch the service of the config given, without waiting for it."""
        self._stopping.clear()
        args = _service_args(config_path)
        self._handle = subprocess.Popen(
            [*self.command, *args], env=env, preexec_fn=_die_with_this_process()
        )
        logger.info("Process ID of %s: %d", self.identity, self._handle.pid)
        threading.Thread(
            target=self._exit_with_binary,
            args=(self._handle,),
            name=f"{self.identity}-watch",
            daemon=True,
        ).start()

    def _exit_with_binary(self, handle: "subprocess.Popen[bytes]") -> None:
        """Exit the actor process when the binary exits unasked, for Ray to restart it.

        Ray only sees the actor: without this, an actor whose binary died stays alive
        and its restart limit never applies.
        """
        exit_code = handle.wait()
        if self._stopping.is_set():
            return
        logger.error(
            "%s (%d) exited with code %d; exiting its actor",
            self.identity,
            handle.pid,
            exit_code,
        )
        logging.shutdown()
        os._exit(1)

    def stop(self) -> None:
        """Stop the binary, killing it on timeout; a no-op when not started."""
        self._stopping.set()
        if self._handle is not None:
            logger.info("Stopping %s (%d)", self.identity, self._handle.pid)
            self._handle.terminate()
            try:
                self._handle.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._handle.kill()
                self._handle.wait()
        self._handle = None

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
