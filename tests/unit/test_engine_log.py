import json

from polars_nanook_ray.engine_log import EngineLog

LICENSE_ERROR = (
    "License file check failed: Failed to read license file: No such file or "
    "directory (os error 2)"
)
BACKTRACE = (
    "\n\nStack backtrace:\n   0: <unknown>\n   1: __libc_start_main_impl\n"
    "             at ./csu/../csu/libc-start.c:360:3\n   2: _start"
)


def _json(level: str, **fields: str) -> str:
    return json.dumps(
        {"fields": fields, "level": level, "timestamp": "2026-11-18T08:38:46Z"}
    )


def _problems(*lines: str) -> list[str]:
    log = EngineLog()
    for line in lines:
        log.add(line)
    return log.problems()


def test_error_message_drops_its_backtrace() -> None:
    line = _json("ERROR", message=LICENSE_ERROR + BACKTRACE, error=LICENSE_ERROR)

    assert _problems(line) == [LICENSE_ERROR]


def test_error_field_follows_a_different_message() -> None:
    line = _json("ERROR", message="Bootstrap failed", error="connection refused")

    assert _problems(line) == ["Bootstrap failed: connection refused"]


def test_error_without_message_is_its_error() -> None:
    assert _problems(_json("ERROR", error="connect to license server")) == [
        "connect to license server"
    ]


def test_info_lines_are_not_problems() -> None:
    assert _problems(_json("INFO", message="Starting boot procedure")) == []


def test_plain_lines_are_errors_without_backtrace_frames() -> None:
    lines = [
        'missing configuration field "shuffle_location" for key `worker`',
        "",
        "Stack backtrace:",
        "   0: <unknown>",
        "             at ./csu/../csu/libc-start.c:360:3",
    ]

    assert _problems(*lines) == [
        'missing configuration field "shuffle_location" for key `worker`'
    ]


def test_quoted_plain_line_is_unquoted() -> None:
    assert _problems(f'"{LICENSE_ERROR}"') == [LICENSE_ERROR]


def test_repeats_and_errors_another_contains_are_dropped() -> None:
    longer = "License server bootstrap failed: connect to license server: refused"
    lines = [
        _json("ERROR", error="connect to license server"),
        _json("ERROR", message=longer + BACKTRACE, error=longer),
        f'"{longer}"',
    ]

    assert _problems(*lines) == [longer]


def test_warnings_stand_in_when_there_are_no_errors() -> None:
    retrying = _json("WARN", message="Retrying", error="Connection refused")

    assert _problems(_json("INFO", message="Starting"), retrying, retrying) == [
        "Retrying: Connection refused"
    ]


def test_warnings_are_left_out_once_there_are_errors() -> None:
    lines = [_json("WARN", message="Retrying"), _json("ERROR", message="Failed")]

    assert _problems(*lines) == ["Failed"]


def test_keeps_the_latest_problems_only() -> None:
    log = EngineLog(max_lines=3)
    for i in range(10):
        log.add(_json("ERROR", message=f"error {i}"))

    assert log.problems() == ["error 7", "error 8", "error 9"]
