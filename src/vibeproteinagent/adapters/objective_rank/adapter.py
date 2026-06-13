from __future__ import annotations

import ast
import hashlib
import json
import math
import operator
import re
import time
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationInfo, field_validator, model_validator

from vibeproteinagent.adapters.base import ToolAdapter, register_adapter, torch
from vibeproteinagent.adapters.context import AdapterRunContext
from vibeproteinagent.core.errors import InvalidParamsError
from vibeproteinagent.workspace import JobStore, Workspace

MissingFormulaPolicy = Literal["fail", "warn", "zero", "ignore"]
MissingGatePolicy = Literal["fail", "warn", "zero", "ignore"]
ScoreDirection = Literal["higher", "lower"]

_PLACEHOLDER_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_.-]*)\}")
_FORMULA_FUNCTION_ARITIES: dict[str, tuple[int | None, int | None]] = {
    "abs": (1, 1),
    "sqrt": (1, 1),
    "log": (1, 2),
    "log10": (1, 1),
    "log2": (1, 1),
    "exp": (1, 1),
    "min": (1, None),
    "max": (1, None),
    "clamp": (3, 3),
}
_REFERENCE_CANDIDATE_TOOLS = frozenset(
    {
        "interpro_sequences",
        "go_sequences",
        "family_profile",
    }
)
_STRUCTURE_KEYS = ("structure_path", "predicted_cif", "predicted_pdb", "pdb_path", "cif_path")
_MATERIAL_SEQUENCE_KEYS = ("sequence", "protein_sequence", "designed_sequence")
_REPORTABLE_STRUCTURE_KEYS = (
    "structure_path",
    "full_structure_path",
    "backbone_path",
    "predicted_cif",
    "predicted_pdb",
    "pdb_path",
    "cif_path",
)
_SCALAR_IDENTITY_KEYS = {
    "candidate_id",
    "name",
    "id",
    "seed",
    "header",
    "source_job_id",
    "source_tool",
    "source_workflow_id",
    "source_workflow_node_id",
}
_SCALAR_VALUE_TYPES = (str, int, float, bool)


class ObjectiveGate(BaseModel):
    id: str = Field(min_length=1)
    metric: str = Field(min_length=1)
    op: Literal["<", "<=", ">", ">=", "==", "!="]
    value: float | None = None
    value_metric: str | None = Field(default=None, min_length=1)
    value_constant: str | None = Field(default=None, min_length=1)
    missing: MissingGatePolicy = "fail"
    description: str | None = None

    @model_validator(mode="after")
    def _one_threshold(self) -> ObjectiveGate:
        choices = [
            self.value is not None,
            self.value_metric is not None,
            self.value_constant is not None,
        ]
        if sum(1 for choice in choices if choice) != 1:
            raise ValueError(
                "gate requires exactly one threshold: value, value_metric, or value_constant"
            )
        return self


class ObjectiveTieBreaker(BaseModel):
    metric: str = Field(min_length=1)
    direction: ScoreDirection = "higher"
    missing: Literal["last", "first"] = "last"
    description: str | None = None


class ObjectiveRankInput(BaseModel):
    job_ids: list[str] = Field(default_factory=list)
    candidates: list[dict[str, Any]] = Field(default_factory=list)
    objective_name: str = Field(min_length=1)
    gates: list[ObjectiveGate] = Field(default_factory=list)
    score_formula: str = Field(min_length=1)
    score_direction: ScoreDirection = "higher"
    constants: dict[str, float] = Field(default_factory=dict)
    missing: MissingFormulaPolicy = "fail"
    formula_missing: dict[str, MissingFormulaPolicy] = Field(default_factory=dict)
    tie_breakers: list[ObjectiveTieBreaker] = Field(default_factory=list)
    top_k: int = Field(10, ge=1, le=1000)
    output_prefix: str = Field("objective_rank", pattern=r"^[A-Za-z0-9_.-]+$")

    @field_validator("candidates", mode="before")
    @classmethod
    def _flatten_nested_candidates(cls, value: Any, info: ValidationInfo) -> list[dict[str, Any]]:
        del info
        rows = _as_list(value)
        if not rows:
            return []
        canonical = [_is_canonical_candidate(row) for row in rows]
        if all(canonical):
            return [_row_from_canonical_candidate(row) for row in rows]
        if any(canonical):
            raise ValueError(
                "Mixed canonical and legacy candidate items are not supported. "
                "Pass either all canonical Candidate[] or all row objects."
            )
        return rows


class ObjectiveRankOutput(BaseModel):
    ranking_intent: str
    objective_name: str
    formula_text: str
    score_direction: ScoreDirection
    gate_summary: list[str]
    candidate_count: int
    ranked_candidates: list[dict[str, Any]]
    ranked_json: str
    ranked_tsv: str
    warnings: list[str] = Field(default_factory=list)
    structured_warnings: list[dict[str, Any]] = Field(default_factory=list)
    runtime_ms: int


def _as_list(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, dict):
        return [value]
    if isinstance(value, list):
        rows: list[dict[str, Any]] = []
        for item in value:
            rows.extend(_as_list(item))
        return rows
    return []


def _is_canonical_candidate(row: dict[str, Any]) -> bool:
    return row.get("schema_version") == "candidate_v1"


def _row_from_canonical_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    source = candidate.get("source") if isinstance(candidate.get("source"), dict) else {}
    producer_tool = str(source.get("tool") or "")
    scores = candidate.get("scores") if isinstance(candidate.get("scores"), dict) else {}
    producer_scores = (
        scores.get(producer_tool)
        if producer_tool and isinstance(scores.get(producer_tool), dict)
        else {}
    )
    row: dict[str, Any] = {
        "candidate_id": candidate.get("candidate_id"),
        "sequence": candidate.get("sequence"),
        "structure_path": candidate.get("structure_path"),
        "full_structure_path": candidate.get("full_structure_path"),
        "backbone_path": candidate.get("backbone_path"),
        "capabilities": candidate.get("capabilities"),
        "source": source,
    }
    if isinstance(producer_scores, dict):
        row.update(producer_scores)
    if scores:
        row["scores"] = dict(scores)
    artifacts = candidate.get("artifacts")
    if isinstance(artifacts, dict) and artifacts:
        row["artifacts"] = dict(artifacts)
    return {key: value for key, value in row.items() if value is not None}


def _candidate_rows_from_output(output: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = output.get("candidates")
    if isinstance(candidates, list) and candidates:
        rows = [item for item in candidates if isinstance(item, dict)]
        if len(rows) == len(candidates) and all(_is_canonical_candidate(row) for row in rows):
            return [_row_from_canonical_candidate(row) for row in rows]
    rows: list[dict[str, Any]] = []
    for key in (
        "ranked_candidates",
        "top_candidates",
        "top_designs",
        "designs",
        "predictions",
        "representative_sequences",
    ):
        rows.extend(item for item in _as_list(output.get(key)) if isinstance(item, dict))
    if not rows and output:
        rows.append(dict(output))
    return rows


def _row_sequence(row: dict[str, Any]) -> str | None:
    for key in _MATERIAL_SEQUENCE_KEYS:
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    return None


def _row_structure_identity(row: dict[str, Any]) -> str | None:
    for key in _STRUCTURE_KEYS:
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    return None


def _candidate_id(row: dict[str, Any], index: int) -> str:
    for key in ("candidate_id", "name", "id", "seed", "header"):
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    sequence = _row_sequence(row)
    if sequence:
        return f"sequence:{sequence[:16]}"
    return str(index + 1)


def _normalise_identity_part(value: Any) -> str:
    if value in (None, ""):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def _material_key(row: dict[str, Any]) -> str:
    source = row.get("source") if isinstance(row.get("source"), dict) else {}
    source_ids = _source_job_ids(row)
    parts = {
        "sequence": _normalise_identity_part(_row_sequence(row)),
        "structure": _normalise_identity_part(_row_structure_identity(row)),
        "source_job_ids": source_ids,
        "source_local_id": _normalise_identity_part(source.get("local_id")),
        "source_native_id": _normalise_identity_part(source.get("native_id")),
        "source_tool": _normalise_identity_part(row.get("source_tool") or source.get("tool")),
    }
    return json.dumps(parts, sort_keys=True, separators=(",", ":"))


def _identity_key(row: dict[str, Any], index: int) -> str:
    keys = _identity_keys(row)
    if keys:
        return "|".join(keys)
    return f"row:{index + 1}"


def _ranked_id_hash(candidate_id: str, material_key: str, source_ids: list[str]) -> str:
    payload = json.dumps(
        {
            "candidate_id": _normalise_identity_part(candidate_id),
            "material_key": material_key,
            "source_ids": source_ids,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8]


def _ranked_identity_metadata(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metadata: list[dict[str, Any]] = []
    groups: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        candidate_id = _candidate_id(row, index)
        material_key = _material_key(row)
        entry = {
            "candidate_id": candidate_id,
            "identity_key": _identity_key(row, index),
            "material_key": material_key,
            "source_job_ids": _source_job_ids(row),
            "ranked_candidate_id": candidate_id,
        }
        metadata.append(entry)
        groups.setdefault(candidate_id, []).append(index)

    for candidate_id, indices in groups.items():
        material_keys = {metadata[index]["material_key"] for index in indices}
        if len(indices) == 1 or len(material_keys) == 1:
            continue
        ordered = sorted(
            indices,
            key=lambda index: (
                metadata[index]["material_key"],
                metadata[index]["identity_key"],
                index,
            ),
        )
        for version, index in enumerate(ordered, start=1):
            material_key = str(metadata[index]["material_key"])
            source_ids = list(metadata[index].get("source_job_ids") or [])
            suffix = _ranked_id_hash(candidate_id, material_key, source_ids)
            metadata[index]["ranked_candidate_id"] = f"{candidate_id}__v{version}_{suffix}"
    return metadata


def _rows_material_compatible(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_sequence = _row_sequence(left)
    right_sequence = _row_sequence(right)
    if left_sequence and right_sequence and left_sequence != right_sequence:
        return False
    left_structure = _row_structure_identity(left)
    right_structure = _row_structure_identity(right)
    if left_structure and right_structure and left_structure != right_structure:
        return False
    return True


def _merge_mapping(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    merged = dict(left)
    for key, value in right.items():
        if value in (None, "", [], {}):
            continue
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            merged[key] = _merge_mapping(current, value)
        elif current in (None, "", [], {}):
            merged[key] = value
        elif current == value:
            continue
        elif key in {"source_job_id", "source_tool"}:
            values = merged.setdefault(f"{key}s", [])
            if isinstance(values, list):
                for item in (current, value):
                    if item not in values:
                        values.append(item)
        else:
            continue
    return merged


def _identity_keys(row: dict[str, Any]) -> list[str]:
    keys: list[str] = []
    for name in ("candidate_id", "name", "id", "header"):
        value = row.get(name)
        if value not in (None, ""):
            keys.append(f"{name}:{value}")
            break
    source = row.get("source") if isinstance(row.get("source"), dict) else {}
    source_job_ids = _source_job_ids(row)
    for name in ("local_id", "native_id"):
        value = source.get(name)
        if value not in (None, "") and source_job_ids:
            for source_job_id in source_job_ids:
                keys.append(f"source_job:{source_job_id}:source.{name}:{value}")
    structure = _row_structure_identity(row)
    if structure:
        keys.append(f"structure:{structure}")
    sequence = _row_sequence(row)
    if sequence:
        keys.append(f"sequence:{sequence}")
    return list(dict.fromkeys(keys))


def _merge_rows_by_identity(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    groups: list[dict[str, Any]] = []
    key_to_group: dict[str, list[int]] = {}
    warnings: list[str] = []
    for row in rows:
        keys = _identity_keys(row)
        matches: list[int] = []
        conflict = False
        for key in keys:
            for group_index in key_to_group.get(key) or []:
                if _rows_material_compatible(groups[group_index], row):
                    if group_index not in matches:
                        matches.append(group_index)
                elif key.startswith("candidate_id:"):
                    conflict = True
        if matches:
            groups[matches[0]] = _merge_mapping(groups[matches[0]], row)
            group_index = matches[0]
        else:
            group_index = len(groups)
            groups.append(dict(row))
            if conflict and keys:
                warnings.append(
                    "objective_rank kept conflicting rows separate for "
                    f"{keys[0]} because sequence or structure differed."
                )
        for key in _identity_keys(groups[group_index]):
            key_to_group.setdefault(key, [])
            if group_index not in key_to_group[key]:
                key_to_group[key].append(group_index)
    return groups, warnings


def _rows_from_jobs(ctx: AdapterRunContext, job_ids: list[str]) -> list[dict[str, Any]]:
    if not job_ids:
        return []
    workspace = Workspace(ctx.workspace_root.name, ctx.workspace_root)
    store = JobStore(workspace)
    rows: list[dict[str, Any]] = []
    for job_id in job_ids:
        job = store.get(job_id)
        if job.tool_name in _REFERENCE_CANDIDATE_TOOLS:
            continue
        output = job.output if isinstance(job.output, dict) else {}
        for row in _candidate_rows_from_output(output):
            row.setdefault("source_job_id", job.job_id)
            row.setdefault("source_tool", job.tool_name)
            if job.workflow_id:
                row.setdefault("source_workflow_id", job.workflow_id)
            if job.workflow_node_id:
                row.setdefault("source_workflow_node_id", job.workflow_node_id)
            rows.append(row)
    return rows


def _coerce_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _flatten_metrics(row: dict[str, Any]) -> dict[str, float]:
    metrics: dict[str, float] = {}

    def visit(prefix: str, value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                part = str(key)
                visit(f"{prefix}.{part}" if prefix else part, child)
            return
        num = _coerce_float(value)
        if num is not None and prefix:
            metrics[prefix] = num

    for key, value in row.items():
        if key in {"scores", "metrics", "confidence_scores"} and isinstance(value, dict):
            visit("", value)
            continue
        num = _coerce_float(value)
        if num is not None:
            metrics[str(key)] = num
    scores = row.get("scores")
    if isinstance(scores, dict):
        visit("", scores)
    metrics_payload = row.get("metrics")
    if isinstance(metrics_payload, dict):
        visit("", metrics_payload)
    confidence = row.get("confidence_scores")
    if isinstance(confidence, dict):
        visit("", confidence)
    return metrics


class _FormulaEvaluator:
    def __init__(self, formula: str) -> None:
        self.formula = formula
        self.paths: list[str] = []
        self.expression = _PLACEHOLDER_RE.sub(self._replace_placeholder, formula)
        try:
            self.tree = ast.parse(self.expression, mode="eval")
        except SyntaxError as exc:
            raise InvalidParamsError(
                "score_formula is not valid objective_rank syntax",
                details={"formula": formula, "error": str(exc)},
            ) from exc
        self._validate(self.tree)

    def _replace_placeholder(self, match: re.Match[str]) -> str:
        path = match.group(1)
        self.paths.append(path)
        return f"__m{len(self.paths) - 1}"

    def _validate(self, node: ast.AST) -> None:
        allowed = (
            ast.Expression,
            ast.BinOp,
            ast.UnaryOp,
            ast.Call,
            ast.Name,
            ast.Load,
            ast.Constant,
            ast.Add,
            ast.Sub,
            ast.Mult,
            ast.Div,
            ast.USub,
            ast.UAdd,
        )
        if not isinstance(node, allowed):
            raise InvalidParamsError(
                "score_formula contains unsupported syntax",
                details={"syntax": type(node).__name__, "formula": self.formula},
            )
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in _FORMULA_FUNCTION_ARITIES:
                raise InvalidParamsError(
                    "score_formula contains an unknown function",
                    details={"formula": self.formula},
                )
            if node.keywords:
                raise InvalidParamsError(
                    "score_formula function calls do not support keyword arguments",
                    details={"function": node.func.id, "formula": self.formula},
                )
            min_args, max_args = _FORMULA_FUNCTION_ARITIES[node.func.id]
            arg_count = len(node.args)
            too_few = min_args is not None and arg_count < min_args
            too_many = max_args is not None and arg_count > max_args
            if too_few or too_many:
                raise InvalidParamsError(
                    "score_formula function called with invalid arity",
                    details={
                        "function": node.func.id,
                        "arg_count": arg_count,
                        "min_args": min_args,
                        "max_args": max_args,
                        "formula": self.formula,
                    },
                )
            for arg in node.args:
                self._validate(arg)
            return
        if isinstance(node, ast.Name):
            if not re.fullmatch(r"__m\d+", node.id):
                raise InvalidParamsError(
                    "score_formula may only reference metrics with ${namespace.metric}",
                    details={"name": node.id, "formula": self.formula},
                )
        if isinstance(node, ast.Constant) and not isinstance(node.value, (int, float)):
            raise InvalidParamsError(
                "score_formula constants must be numeric",
                details={"formula": self.formula},
            )
        for child in ast.iter_child_nodes(node):
            self._validate(child)

    def evaluate(
        self,
        metrics: dict[str, float],
        constants: dict[str, float],
        missing: MissingFormulaPolicy,
        formula_missing: dict[str, MissingFormulaPolicy],
    ) -> tuple[float | None, list[dict[str, Any]], list[str]]:
        values: dict[str, float] = {}
        components: list[dict[str, Any]] = []
        warnings: list[str] = []
        failed = False
        for index, path in enumerate(self.paths):
            var = f"__m{index}"
            value = _metric_path_value(path, metrics, constants)
            policy = formula_missing.get(path, missing)
            component: dict[str, Any] = {
                "metric": path,
                "value": value,
                "missing_policy": policy,
            }
            if value is None:
                component["missing"] = True
                if policy == "fail":
                    failed = True
                    warnings.append(f"score metric missing: {path}")
                elif policy == "warn":
                    warnings.append(f"score metric missing, using 0: {path}")
                    value = 0.0
                elif policy == "zero":
                    value = 0.0
                elif policy == "ignore":
                    value = 0.0
            if value is not None:
                values[var] = value
                component["used_value"] = value
            components.append(component)
        if failed:
            return None, components, warnings
        try:
            result = _eval_ast(self.tree.body, values)
        except ZeroDivisionError as exc:
            return None, components, [*warnings, f"score_formula division by zero: {exc}"]
        except (OverflowError, ValueError) as exc:
            return None, components, [*warnings, f"score_formula evaluation error: {exc}"]
        return float(result), components, warnings


def _eval_ast(node: ast.AST, variables: dict[str, float]) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.Name):
        return variables[node.id]
    if isinstance(node, ast.UnaryOp):
        value = _eval_ast(node.operand, variables)
        if isinstance(node.op, ast.USub):
            return -value
        if isinstance(node.op, ast.UAdd):
            return value
    if isinstance(node, ast.BinOp):
        ops: dict[type[ast.operator], Any] = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
        }
        fn = ops.get(type(node.op))
        if fn is not None:
            return float(fn(_eval_ast(node.left, variables), _eval_ast(node.right, variables)))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        args = [_eval_ast(arg, variables) for arg in node.args]
        function = node.func.id
        if function == "min":
            return float(min(args))
        if function == "max":
            return float(max(args))
        if function == "abs":
            return abs(args[0])
        if function == "sqrt":
            return math.sqrt(args[0])
        if function == "log":
            return math.log(args[0]) if len(args) == 1 else math.log(args[0], args[1])
        if function == "log10":
            return math.log10(args[0])
        if function == "log2":
            return math.log2(args[0])
        if function == "exp":
            return math.exp(args[0])
        if function == "clamp":
            value, low, high = args
            return float(max(low, min(high, value)))
    raise InvalidParamsError("score_formula could not be evaluated")


def _metric_path_value(
    path: str,
    metrics: dict[str, float],
    constants: dict[str, float],
) -> float | None:
    if path.startswith("constants."):
        return _coerce_float(constants.get(path.split(".", 1)[1]))
    if path in metrics:
        return metrics[path]
    # Bare metric aliases remain useful when upstream rows are flat.
    if "." in path:
        bare = path.rsplit(".", 1)[1]
        return metrics.get(bare)
    return None


_GATE_OPS = {
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
    "==": operator.eq,
    "!=": operator.ne,
}


def _gate_threshold(
    gate: ObjectiveGate,
    metrics: dict[str, float],
    constants: dict[str, float],
) -> float | None:
    if gate.value is not None:
        return gate.value
    if gate.value_metric:
        return _metric_path_value(gate.value_metric, metrics, constants)
    if gate.value_constant:
        return _coerce_float(constants.get(gate.value_constant))
    return None


def _evaluate_gates(
    gates: list[ObjectiveGate],
    metrics: dict[str, float],
    constants: dict[str, float],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    passed: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    warnings: list[str] = []
    for gate in gates:
        value = _metric_path_value(gate.metric, metrics, constants)
        threshold = _gate_threshold(gate, metrics, constants)
        threshold_ref = (
            gate.value_metric
            if gate.value_metric
            else gate.value_constant
            if gate.value_constant
            else gate.value
        )
        record = {
            "id": gate.id,
            "metric": gate.metric,
            "op": gate.op,
            "value": value,
            "threshold": threshold,
            "threshold_ref": threshold_ref,
            "description": gate.description,
            "missing_policy": gate.missing,
        }
        if value is None or threshold is None:
            record["missing"] = True
            record["reason"] = "missing_metric"
            if gate.missing == "fail":
                failed.append(record)
            elif gate.missing == "warn":
                warnings.append(f"gate metric missing: {gate.id}")
                passed.append({**record, "warning": "missing_metric"})
            elif gate.missing == "zero":
                used_value = 0.0 if value is None else value
                used_threshold = 0.0 if threshold is None else threshold
                zero_record = {
                    **record,
                    "value": value,
                    "threshold": threshold,
                    "used_value": used_value,
                    "used_threshold": used_threshold,
                }
                if _GATE_OPS[gate.op](used_value, used_threshold):
                    passed.append(zero_record)
                else:
                    failed.append({**zero_record, "reason": "condition_not_met"})
            continue
        if _GATE_OPS[gate.op](value, threshold):
            passed.append(record)
        else:
            record["reason"] = "condition_not_met"
            failed.append(record)
    return passed, failed, warnings


def _gate_summary(gates: list[ObjectiveGate]) -> list[str]:
    summary: list[str] = []
    for gate in gates:
        if gate.value is not None:
            rhs = _format_number(gate.value)
        elif gate.value_metric:
            rhs = gate.value_metric
        else:
            rhs = f"constants.{gate.value_constant}"
        summary.append(f"{gate.metric} {gate.op} {rhs}")
    return summary


def _format_number(value: Any) -> str:
    num = _coerce_float(value)
    if num is None:
        return str(value)
    if abs(num - round(num)) < 1e-9:
        return str(round(num))
    return f"{num:.6g}"


def _tie_breaker_values(
    tie_breakers: list[ObjectiveTieBreaker],
    metrics: dict[str, float],
    constants: dict[str, float],
) -> dict[str, float | None]:
    return {tb.metric: _metric_path_value(tb.metric, metrics, constants) for tb in tie_breakers}


def _sort_key(
    row: dict[str, Any],
    params: ObjectiveRankInput,
) -> tuple[Any, ...]:
    failed_count = len(row.get("failed_gates") or [])
    score = _coerce_float(row.get("objective_score"))
    if score is None:
        score_key = float("inf")
    elif params.score_direction == "higher":
        score_key = -score
    else:
        score_key = score
    values = (
        row.get("tie_breaker_values") if isinstance(row.get("tie_breaker_values"), dict) else {}
    )
    tb_keys: list[float] = []
    for tb in params.tie_breakers:
        value = _coerce_float(values.get(tb.metric))
        if value is None:
            tb_keys.append(float("-inf") if tb.missing == "first" else float("inf"))
        elif tb.direction == "higher":
            tb_keys.append(-value)
        else:
            tb_keys.append(value)
    return (
        failed_count,
        score_key,
        *tb_keys,
        str(row.get("ranked_candidate_id") or row.get("candidate_id") or ""),
    )


def _copy_original_fields(row: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in (
        "sequence",
        "protein_sequence",
        "designed_sequence",
        "structure_path",
        "full_structure_path",
        "backbone_path",
        "scores",
        "artifacts",
        "source",
        "source_job_id",
        "source_tool",
        "source_workflow_id",
        "source_workflow_node_id",
        "capabilities",
    ):
        if key in row and row[key] not in (None, "", [], {}):
            out[key] = row[key]
    return out


def _source_identity(row: dict[str, Any]) -> dict[str, Any]:
    source = row.get("source") if isinstance(row.get("source"), dict) else {}
    identity: dict[str, Any] = {}
    for key in ("tool", "node_id", "job_id", "local_id", "native_id", "index"):
        value = source.get(key)
        if value not in (None, "", [], {}):
            identity[key] = value
    source_job_ids = _source_job_ids(row)
    if source_job_ids:
        identity["job_ids"] = source_job_ids
    for row_key, identity_key in (
        ("source_tool", "tool"),
        ("source_job_id", "job_id"),
        ("source_workflow_id", "workflow_id"),
        ("source_workflow_node_id", "workflow_node_id"),
    ):
        value = row.get(row_key)
        if value not in (None, "", [], {}) and identity_key not in identity:
            identity[identity_key] = value
    return identity


def _material_identity(row: dict[str, Any]) -> dict[str, Any]:
    artifacts = row.get("artifacts") if isinstance(row.get("artifacts"), dict) else {}
    material: dict[str, Any] = {
        "has_sequence": _row_sequence(row) is not None,
        "has_structure": _row_structure_identity(row) is not None
        or any(row.get(key) not in (None, "") for key in _REPORTABLE_STRUCTURE_KEYS),
        "artifact_namespaces": sorted(str(key) for key in artifacts),
    }
    sequence = _row_sequence(row)
    if sequence:
        material["sequence"] = sequence
    structure = _row_structure_identity(row)
    if structure:
        material["structure"] = structure
    for key in ("structure_path", "full_structure_path", "backbone_path"):
        value = row.get(key)
        if value not in (None, ""):
            material[key] = value
    return material


def _is_scalar_only_row(row: dict[str, Any]) -> bool:
    populated = {key: value for key, value in row.items() if value not in (None, "", [], {})}
    if not populated:
        return True
    for key, value in populated.items():
        if key in _SCALAR_IDENTITY_KEYS:
            continue
        if not isinstance(value, _SCALAR_VALUE_TYPES):
            return False
    return True


def _contains_numeric_leaf(value: Any) -> bool:
    if isinstance(value, dict):
        return any(_contains_numeric_leaf(child) for child in value.values())
    if isinstance(value, list):
        return any(_contains_numeric_leaf(child) for child in value)
    return _coerce_float(value) is not None


def _score_evidence_namespaces(row: dict[str, Any]) -> list[str]:
    namespaces: set[str] = set()
    for container_key in ("scores", "metrics", "confidence_scores"):
        container = row.get(container_key)
        if not isinstance(container, dict):
            continue
        for key, value in container.items():
            namespace = str(key)
            if namespace == "objective_rank":
                continue
            if isinstance(value, dict):
                if _contains_numeric_leaf(value):
                    namespaces.add(namespace)
            elif "." in namespace and _contains_numeric_leaf(value):
                namespaces.add(namespace.split(".", 1)[0])
    for key, value in row.items():
        if key in {"scores", "metrics", "confidence_scores"}:
            continue
        if "." in str(key) and _contains_numeric_leaf(value):
            namespaces.add(str(key).split(".", 1)[0])
    return sorted(namespaces)


def _has_reportable_material_or_artifacts(row: dict[str, Any]) -> bool:
    if _row_sequence(row):
        return True
    if any(row.get(key) not in (None, "", [], {}) for key in _REPORTABLE_STRUCTURE_KEYS):
        return True
    artifacts = row.get("artifacts")
    return isinstance(artifacts, dict) and any(
        value not in (None, "", [], {}) for value in artifacts.values()
    )


def _identity_contract_warnings(
    row: dict[str, Any],
    *,
    candidate_id: str,
    ranked_candidate_id: str,
) -> list[dict[str, Any]]:
    warnings: list[dict[str, Any]] = []
    if _is_scalar_only_row(row):
        warnings.append(
            {
                "code": "scalar_only_ranked_row",
                "severity": "warning",
                "message": (
                    "Ranked row contains only scalar fields; no source material or "
                    "reportable evidence can be attached."
                ),
                "candidate_id": candidate_id,
                "ranked_candidate_id": ranked_candidate_id,
                "present_fields": sorted(
                    str(key) for key, value in row.items() if value not in (None, "", [], {})
                ),
                "required_any_of": [
                    "sequence",
                    "structure_path",
                    "full_structure_path",
                    "backbone_path",
                    "artifacts",
                    "source",
                ],
            }
        )

    score_namespaces = _score_evidence_namespaces(row)
    if score_namespaces and not _has_reportable_material_or_artifacts(row):
        warnings.append(
            {
                "code": "missing_reportable_evidence",
                "severity": "warning",
                "message": (
                    "Ranked row has namespaced scores but no reportable sequence, "
                    "structure path, or artifacts."
                ),
                "candidate_id": candidate_id,
                "ranked_candidate_id": ranked_candidate_id,
                "score_namespaces": score_namespaces,
                "required_any_of": [
                    "sequence",
                    "structure_path",
                    "full_structure_path",
                    "backbone_path",
                    "artifacts",
                ],
            }
        )
    return warnings


@register_adapter("descriptor.yaml")
class ObjectiveRankAdapter(ToolAdapter):
    def __init__(self, config: dict, weights_dir: Path, device: torch.device) -> None:
        self.device = device

    def run(self, params: ObjectiveRankInput, ctx: AdapterRunContext) -> ObjectiveRankOutput:
        t0 = time.time()
        evaluator = _FormulaEvaluator(params.score_formula)
        rows = [*_rows_from_jobs(ctx, params.job_ids), *_as_list(params.candidates)]
        rows, warnings = _merge_rows_by_identity(rows)
        if not rows:
            raise ValueError("objective_rank requires at least one candidate row or job_id output")
        identity_metadata = _ranked_identity_metadata(rows)
        gate_summary = _gate_summary(params.gates)

        ranked: list[dict[str, Any]] = []
        structured_warnings: list[dict[str, Any]] = []
        for index, row in enumerate(rows):
            identity = identity_metadata[index]
            metrics = _flatten_metrics(row)
            objective_score, components, score_warnings = evaluator.evaluate(
                metrics,
                params.constants,
                params.missing,
                params.formula_missing,
            )
            passed_gates, failed_gates, gate_warnings = _evaluate_gates(
                params.gates,
                metrics,
                params.constants,
            )
            candidate_warnings = [*score_warnings, *gate_warnings]
            candidate_id = str(identity["candidate_id"])
            ranked_candidate_id = str(identity["ranked_candidate_id"])
            contract_warnings = _identity_contract_warnings(
                row,
                candidate_id=candidate_id,
                ranked_candidate_id=ranked_candidate_id,
            )
            tie_values = _tie_breaker_values(params.tie_breakers, metrics, params.constants)
            objective_block = {
                "objective_name": params.objective_name,
                "objective_score": (
                    round(objective_score, 6) if objective_score is not None else None
                ),
                "score_formula": params.score_formula,
                "formula_text": params.score_formula,
                "score_direction": params.score_direction,
                "gate_summary": gate_summary,
                "passed_gates": passed_gates,
                "failed_gates": failed_gates,
                "score_components": components,
                "tie_breaker_values": tie_values,
                "ranked_candidate_id": ranked_candidate_id,
            }
            scores = dict(row.get("scores") or {}) if isinstance(row.get("scores"), dict) else {}
            scores["objective_rank"] = {
                "rank": 0,
                **objective_block,
            }
            ranked.append(
                {
                    "rank": 0,
                    "candidate_id": candidate_id,
                    "ranked_candidate_id": ranked_candidate_id,
                    "identity_key": identity["identity_key"],
                    "material_key": identity["material_key"],
                    "score": objective_block["objective_score"],
                    "ranking_intent": "objective",
                    **objective_block,
                    "warnings": candidate_warnings,
                    "structured_warnings": contract_warnings,
                    "candidate_ref": {
                        "candidate_id": candidate_id,
                        "ranked_candidate_id": ranked_candidate_id,
                        "identity_key": identity["identity_key"],
                        "material_key": identity["material_key"],
                        "source_job_ids": identity["source_job_ids"],
                        "source": _source_identity(row),
                        "material_identity": _material_identity(row),
                    },
                    **_copy_original_fields(row),
                    "scores": scores,
                }
            )

        ranked.sort(key=lambda item: _sort_key(item, params))
        for rank, row in enumerate(ranked, start=1):
            row["rank"] = rank
            objective_scores = row.get("scores")
            if isinstance(objective_scores, dict):
                block = objective_scores.get("objective_rank")
                if isinstance(block, dict):
                    block["rank"] = rank
        selected = ranked[: params.top_k]
        for row in selected:
            warnings.extend(
                f"{row['candidate_id']}: {warning}"
                for warning in row.get("warnings", [])
                if isinstance(warning, str)
            )
            structured_warnings.extend(
                warning
                for warning in row.get("structured_warnings", [])
                if isinstance(warning, dict)
            )

        out_dir = ctx.job_dir / "outputs"
        out_dir.mkdir(parents=True, exist_ok=True)
        json_path = out_dir / f"{params.output_prefix}.json"
        tsv_path = out_dir / f"{params.output_prefix}.tsv"
        json_path.write_text(json.dumps(selected, indent=2, sort_keys=True) + "\n")
        with tsv_path.open("w") as fh:
            fh.write(
                "rank\tcandidate_id\tranked_candidate_id\tobjective_score\t"
                "passed_gates\tfailed_gates\n"
            )
            for row in selected:
                objective_score = (
                    row.get("objective_score") if row.get("objective_score") is not None else ""
                )
                fh.write(
                    f"{row['rank']}\t{row['candidate_id']}\t{row['ranked_candidate_id']}\t"
                    f"{objective_score}\t"
                    f"{len(row.get('passed_gates') or [])}\t"
                    f"{len(row.get('failed_gates') or [])}\n"
                )
        ctx.emit_progress(1.0)
        return ObjectiveRankOutput(
            ranking_intent="objective",
            objective_name=params.objective_name,
            formula_text=params.score_formula,
            score_direction=params.score_direction,
            gate_summary=gate_summary,
            candidate_count=len(ranked),
            ranked_candidates=selected,
            ranked_json=str(json_path),
            ranked_tsv=str(tsv_path),
            warnings=sorted(dict.fromkeys(warnings)),
            structured_warnings=structured_warnings,
            runtime_ms=int((time.time() - t0) * 1000),
        )

    def estimate_vram(self, params: ObjectiveRankInput) -> int:
        return 0


def _source_job_ids(row: dict[str, Any]) -> list[str]:
    values: list[str] = []
    source = row.get("source") if isinstance(row.get("source"), dict) else {}
    source_job_id = source.get("job_id")
    if isinstance(source_job_id, str) and source_job_id:
        values.append(source_job_id)
    for key in ("source_job_id", "source_job_ids", "evidence_job_ids"):
        value = row.get(key)
        if isinstance(value, str) and value:
            values.append(value)
        elif isinstance(value, list):
            values.extend(str(item) for item in value if item)
    return sorted(dict.fromkeys(values))


__all__ = ["ObjectiveRankAdapter", "ObjectiveRankInput", "ObjectiveRankOutput"]
