from __future__ import annotations

import re
from pathlib import Path

from vibeproteinagent.core.log import (
    Logger,
    format_record,
    get_file_logger,
)

TS_RE = re.compile(r"^\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}\]")


def test_format_record_has_iso_timestamp_and_fixed_width_level() -> None:
    line = format_record("supervisor", "info", "hello")
    assert TS_RE.match(line), line
    assert " [INFO " in line or " [INFO ]" in line or " [INFO]" in line
    # Component column is left-aligned and bracketed.
    assert "[supervisor]" in line
    assert line.endswith("hello")


def test_format_record_emits_kv_pairs_in_declared_order() -> None:
    line = format_record(
        "worker:2",
        "info",
        "adapter_load_done",
        {"name": "proteinmpnn", "vram_mib": 14, "t_ms": 1820},
    )
    assert "name=proteinmpnn" in line
    assert "vram_mib=14" in line
    assert "t_ms=1820" in line
    # Order preserved.
    assert line.index("name=") < line.index("vram_mib=") < line.index("t_ms=")


def test_format_record_value_coercion() -> None:
    line = format_record(
        "x",
        "warn",
        "msg",
        {
            "s": "abc",
            "b_true": True,
            "b_false": False,
            "n": None,
            "i": 7,
            "f": 1.5,
            "with space": "a b",
        },
    )
    assert "s=abc" in line
    assert "b_true=true" in line
    assert "b_false=false" in line
    assert "n=-" in line
    assert "i=7" in line
    assert "f=1.5" in line
    # Quoting for spaces.
    assert '"a b"' in line


def test_logger_writes_to_file(tmp_path: Path) -> None:
    log_path = tmp_path / "x.log"
    log = Logger("comp", log_path)
    log.info("ready", socket="/tmp/x.sock", pid=42)
    log.warn("flaky", count=3)
    log.error("boom", code="E_INTERNAL")
    log.debug("verbose_detail", x=1)
    log.close()
    text = log_path.read_text().strip().splitlines()
    assert len(text) == 4
    assert "[INFO" in text[0] and "ready" in text[0] and "pid=42" in text[0]
    assert "[WARN" in text[1] and "count=3" in text[1]
    assert "[ERROR" in text[2] and "code=E_INTERNAL" in text[2]
    assert "[DEBUG" in text[3]


def test_get_file_logger_creates_parent_dirs(tmp_path: Path) -> None:
    nested = tmp_path / "a" / "b" / "c" / "log.txt"
    log = get_file_logger("comp", nested)
    log.info("hello")
    log.close()
    assert nested.is_file()
    assert "hello" in nested.read_text()


def test_logger_swallows_write_errors_after_close(tmp_path: Path) -> None:
    log = Logger("comp", tmp_path / "x.log")
    log.close()
    # Must not raise.
    log.info("after close")
