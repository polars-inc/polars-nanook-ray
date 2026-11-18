import collections
import json
import re
import threading

# a frame of a Rust or Python backtrace, or the line opening one
_BACKTRACE = re.compile(r"^(\s|Stack backtrace:|Traceback \(most recent call last\):)")


class EngineLog:
    """The latest errors and warnings an engine wrote, to tell why it fails to start.

    The engine logs JSON lines, and writes its fatal error as a plain line before it
    exits; with `RUST_BACKTRACE` set, both carry a backtrace, which is left out.
    """

    def __init__(self, max_lines: int = 100) -> None:
        self._lines: collections.deque[tuple[str, str]] = collections.deque(
            maxlen=max_lines
        )
        self._lock = threading.Lock()

    def add(self, line: str) -> None:
        """Keep `line` if it is an error or a warning."""
        if (problem := _parse(line)) is not None:
            with self._lock:
                self._lines.append(problem)

    def problems(self) -> list[str]:
        """Return the errors kept, or the warnings if none, oldest first.

        Leaves out repeats, and errors contained in another, such as the cause the
        engine logs before the error it leads to.
        """
        with self._lock:
            lines = list(self._lines)
        errors = [text for level, text in lines if level == "ERROR"]
        kept: list[str] = []
        for text in errors or [text for _, text in lines]:
            if any(text in other for other in kept):
                continue
            kept = [other for other in kept if other not in text]
            kept.append(text)
        return kept


def _parse(line: str) -> tuple[str, str] | None:
    """Return the level and text of an error or warning line, else None."""
    line = line.rstrip()
    try:
        record = json.loads(line)
    except ValueError:
        record = None
    if isinstance(record, dict):
        level = record.get("level")
        fields = record.get("fields") or {}
        if level not in ("ERROR", "WARN") or not isinstance(fields, dict):
            return None
        message = str(fields.get("message", "")).split("\n\nStack backtrace:")[0]
        error = str(fields.get("error", ""))
        if not message or error.startswith(message):
            text = error or message
        elif error and error not in message:
            text = f"{message}: {error}"
        else:
            text = message
        return (level, text.strip()) if text.strip() else None

    if not line or _BACKTRACE.match(line):
        return None
    if len(line) > 1 and line[0] == line[-1] == '"':
        line = line[1:-1]
    return "ERROR", line
