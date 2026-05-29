from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path


def _summary_python_block() -> str:
    script = Path("validation_runs/run_live_provider_matrix_vm024.sh").read_text()
    for block in re.findall(r"<<'PY'\n(.*?)\nPY", script, flags=re.S):
        if "case_counts =" in block and "provider_counts =" in block:
            return block
    raise AssertionError("live matrix summary Python block not found")


def test_live_matrix_summary_separates_case_and_provider_counts(tmp_path: Path) -> None:
    run_root = tmp_path / "matrix"
    run_root.mkdir()
    status_tsv = run_root / "status.tsv"
    summary_md = run_root / "RUN_SUMMARY.md"
    summary_json = run_root / "summary.json"
    guardrail_json = run_root / "release_evidence_guardrail.json"
    expected = run_root / "expected_cases.tsv"
    fail_output = run_root / "actual_binder_qwen.json"
    fail_output.write_text('{"ok": true, "data": {"state": "failed"}}\n')
    (run_root / "matrix.log").write_text("clean matrix log\n")
    status_tsv.write_text(
        "\t".join(
            [
                "case",
                "mode",
                "provider",
                "model",
                "status",
                "case_status",
                "process_exit_code",
                "api_ok",
                "run_state",
                "validation_status",
                "seconds",
                "output",
            ]
        )
        + "\n"
        + "\n".join(
            [
                "\t".join(
                    [
                        "actual_semantic_qwen",
                        "actual",
                        "openrouter",
                        "qwen",
                        "PASS",
                        "PASS",
                        "0",
                        "true",
                        "completed",
                        "PASS",
                        "1.0",
                        str(run_root / "actual_semantic_qwen.json"),
                    ]
                ),
                "\t".join(
                    [
                        "actual_binder_qwen",
                        "actual",
                        "openrouter",
                        "qwen",
                        "PASS",
                        "FAIL",
                        "0",
                        "true",
                        "failed",
                        "FAIL",
                        "2.0",
                        str(fail_output),
                    ]
                ),
                "\t".join(
                    [
                        "actual_semantic_codex",
                        "actual",
                        "codex-cli",
                        "codex",
                        "PASS",
                        "PASS",
                        "0",
                        "true",
                        "completed",
                        "PASS",
                        "3.0",
                        str(run_root / "actual_semantic_codex.json"),
                    ]
                ),
            ]
        )
        + "\n"
    )
    expected.write_text(
        "case\tmode\tprovider\tmodel\n"
        "actual_semantic_qwen\tactual\topenrouter\tqwen\n"
        "actual_binder_qwen\tactual\topenrouter\tqwen\n"
        "actual_semantic_codex\tactual\tcodex-cli\tcodex\n"
        "actual_binder_codex\tactual\tcodex-cli\tcodex\n"
    )

    subprocess.run(
        [
            sys.executable,
            "-c",
            _summary_python_block(),
            str(status_tsv),
            str(summary_md),
            str(summary_json),
            str(run_root),
            "0",
            "validation_runs/run_live_provider_matrix_vm024.sh",
            "sha-start",
            "sha-start",
            "true",
            str(guardrail_json),
        ],
        check=True,
        cwd=Path.cwd(),
    )

    summary = json.loads(summary_json.read_text())
    assert summary["case_counts"] == {
        "PASS": 2,
        "FAIL": 1,
        "BLOCKED": 0,
        "MISSING": 1,
    }
    assert summary["provider_counts"] == {
        "PASS": 1,
        "FAIL": 1,
        "BLOCKED": 0,
        "MISSING": 0,
    }
    assert summary["counts"] == summary["case_counts"]
    assert summary["release_evidence"]["release_evidence_valid"] is True
    assert summary["release_evidence"]["log_guardrail_violations"] == []
    assert json.loads(guardrail_json.read_text())["release_evidence_valid"] is True
    assert "1 failed" in summary_md.read_text()


def test_live_matrix_summary_marks_shell_log_integrity_errors_not_release_evidence(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "matrix"
    run_root.mkdir()
    status_tsv = run_root / "status.tsv"
    summary_md = run_root / "RUN_SUMMARY.md"
    summary_json = run_root / "summary.json"
    guardrail_json = run_root / "release_evidence_guardrail.json"
    expected = run_root / "expected_cases.tsv"
    output = run_root / "actual_semantic_qwen.json"
    output.write_text('{"ok": true, "data": {"state": "completed"}}\n')
    status_tsv.write_text(
        "\t".join(
            [
                "case",
                "mode",
                "provider",
                "model",
                "status",
                "case_status",
                "process_exit_code",
                "api_ok",
                "run_state",
                "validation_status",
                "seconds",
                "output",
            ]
        )
        + "\n"
        + "\t".join(
            [
                "actual_semantic_qwen",
                "actual",
                "openrouter",
                "qwen",
                "PASS",
                "PASS",
                "0",
                "true",
                "completed",
                "PASS",
                "1.0",
                str(output),
            ]
        )
        + "\n"
    )
    expected.write_text("case\tmode\tprovider\tmodel\nactual_semantic_qwen\tactual\topenrouter\tqwen\n")
    (run_root / "matrix.log").write_text(
        "validation_runs/run_live_provider_matrix_vm024.sh: line 623: ases,: command not found\n"
    )

    subprocess.run(
        [
            sys.executable,
            "-c",
            _summary_python_block(),
            str(status_tsv),
            str(summary_md),
            str(summary_json),
            str(run_root),
            "0",
            "validation_runs/run_live_provider_matrix_vm024.sh",
            "sha-start",
            "sha-start",
            "true",
            str(guardrail_json),
        ],
        check=True,
        cwd=Path.cwd(),
    )

    summary = json.loads(summary_json.read_text())
    assert summary["case_counts"]["PASS"] == 1
    release = summary["release_evidence"]
    assert release["release_evidence_valid"] is False
    assert release["log_guardrail_violations"][0]["pattern"] == "command not found"
    assert json.loads(guardrail_json.read_text())["release_evidence_valid"] is False


def test_live_matrix_script_keeps_clean_run_guardrails() -> None:
    script = Path("validation_runs/run_live_provider_matrix_vm024.sh").read_text()

    subprocess.run(
        ["bash", "-n", "validation_runs/run_live_provider_matrix_vm024.sh"],
        check=True,
        cwd=Path.cwd(),
    )
    assert "bash -n \"${SCRIPT_PATH}\"" in script
    assert "case_status" in script
    assert "output directory is not empty; use a clean run directory" in script
    assert "output directory already contains matrix result files" in script
    assert 'final_submission.get("outcome") not in {"accepted", "no_acceptable_candidate"}' in script
    assert 'result.get("technical_failure") is not False' in script
