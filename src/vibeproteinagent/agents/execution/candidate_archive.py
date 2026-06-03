from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict, dataclass
from dataclasses import field as dataclass_field
from typing import Any, Literal

from jsonschema.exceptions import ValidationError

from vibeproteinagent.agents.planning.candidate import validate_candidate_v1
from vibeproteinagent.agents.planning.candidate_merge import (
    merge_candidate_records,
    summarize_candidate_merge_audit,
)
from vibeproteinagent.workflows.store import WorkflowStore
from vibeproteinagent.workspace import Workspace

_CANONICAL_CANDIDATE_CONTAINERS = ("candidates", "ranked_candidates")
_FALLBACK_CANDIDATE_CONTAINERS = ("representative_sequences",)
_MATERIAL_CAPABILITIES = {
    "has_sequence",
    "has_backbone_structure",
    "has_full_structure",
    "has_candidate_scores",
}


@dataclass(frozen=True)
class CandidateRecord:
    candidate_id: str
    version_id: str
    round_index: int
    workflow_id: str | None
    node_id: str | None
    job_id: str | None
    status: Literal["current", "survivor", "rejected"]
    candidate: dict[str, Any]
    raw_source: dict[str, Any]
    validation_errors: list[dict[str, Any]]
    run_id: str | None = None


@dataclass(frozen=True)
class CandidateSnapshotRef:
    candidate_id: str
    version_id: str


@dataclass(frozen=True)
class CandidateSnapshot:
    snapshot_id: str
    round_index: int
    kind: Literal["current_round", "survivor"]
    candidate_refs: list[CandidateSnapshotRef]
    merge_audit: dict[str, Any]
    run_id: str | None = None
    workflow_ids: list[str] = dataclass_field(default_factory=list)

    @property
    def candidate_ids(self) -> list[str]:
        return [ref.candidate_id for ref in self.candidate_refs]


class CandidateArchive:
    """Typed, validated candidate source for executor materialization."""

    def __init__(self, workspace: Workspace, *, session_store: Any | None = None) -> None:
        self.workspace = workspace
        self.session_store = session_store
        self.records_by_version: dict[str, CandidateRecord] = {}
        self.snapshots: dict[str, CandidateSnapshot] = {}

    def ingest_workflow_round(
        self,
        *,
        run_id: str,
        round_index: int,
        workflow_id: str | None,
        raw_workflow_summary: dict[str, Any] | None = None,
    ) -> CandidateSnapshot:
        summary = (
            raw_workflow_summary
            if isinstance(raw_workflow_summary, dict) and raw_workflow_summary.get("nodes")
            else self._raw_workflow_summary(workflow_id)
        )
        raw_items = _raw_candidate_items(summary)
        valid_candidates: list[dict[str, Any]] = []
        valid_source_by_id: dict[str, dict[str, Any]] = {}
        rejected_records: list[CandidateRecord] = []

        for position, item in enumerate(raw_items):
            candidate = item.get("candidate")
            seed_errors = item.get("candidate_errors")
            errors = list(seed_errors) if isinstance(seed_errors, list) else []
            errors.extend(_candidate_validation_errors(candidate))
            if errors:
                rejected_records.append(
                    self._record(
                        run_id=run_id,
                        round_index=round_index,
                        workflow_id=workflow_id,
                        status="rejected",
                        candidate=candidate if isinstance(candidate, dict) else {},
                        raw_source={**item, "position": position},
                        validation_errors=errors,
                    )
                )
                continue
            assert isinstance(candidate, dict)
            valid_candidates.append(candidate)
            valid_source_by_id[candidate["candidate_id"]] = {**item, "position": position}

        merge = merge_candidate_records(
            valid_candidates,
            strip_private_keys=True,
            include_private_positions=False,
        )
        current_records: list[CandidateRecord] = []
        for candidate in merge.candidates:
            source = valid_source_by_id.get(str(candidate.get("candidate_id")), {})
            current_records.append(
                self._record(
                    run_id=run_id,
                    round_index=round_index,
                    workflow_id=workflow_id,
                    status="current",
                    candidate=candidate,
                    raw_source=source,
                    validation_errors=[],
                )
            )

        for record in [*rejected_records, *current_records]:
            self._save_record(run_id, record)
        current_refs = [
            CandidateSnapshotRef(
                candidate_id=record.candidate_id,
                version_id=record.version_id,
            )
            for record in current_records
        ]

        snapshot = CandidateSnapshot(
            snapshot_id=_snapshot_id(run_id, round_index, "current_round", current_refs),
            round_index=round_index,
            kind="current_round",
            candidate_refs=current_refs,
            merge_audit=merge.audit,
            run_id=run_id,
            workflow_ids=_workflow_ids([workflow_id]),
        )
        self._save_snapshot(run_id, snapshot)
        return snapshot

    def merge_survivors(
        self,
        *,
        run_id: str,
        round_index: int,
        previous_snapshot_id: str | None,
        current_snapshot_id: str | None,
    ) -> CandidateSnapshot:
        previous_snapshot = self._snapshot(previous_snapshot_id)
        current_snapshot = self._snapshot(current_snapshot_id)
        source_records = [
            record
            for record in [
                *self._records_from_snapshot(previous_snapshot),
                *self._records_from_snapshot(current_snapshot),
            ]
            if record.status != "rejected"
        ]
        merge = merge_candidate_records(
            [record.candidate for record in source_records],
            strip_private_keys=False,
            include_private_positions=True,
        )
        survivor_refs: list[CandidateSnapshotRef] = []
        for candidate in merge.candidates:
            representative_record = _representative_record_for_merged_candidate(
                candidate,
                source_records,
            )
            source_lineage = _source_records_for_merged_candidate(candidate, source_records)
            saved_candidate = _strip_candidate_merge_private(candidate)
            record = self._record(
                run_id=run_id,
                round_index=round_index,
                workflow_id=representative_record.workflow_id if representative_record else None,
                status="survivor",
                candidate=saved_candidate,
                raw_source=_survivor_raw_source(
                    previous_snapshot=previous_snapshot,
                    current_snapshot=current_snapshot,
                    representative_record=representative_record,
                    source_records=source_lineage,
                ),
                validation_errors=[],
            )
            self._save_record(run_id, record)
            survivor_refs.append(
                CandidateSnapshotRef(
                    candidate_id=record.candidate_id,
                    version_id=record.version_id,
                )
            )
        snapshot = CandidateSnapshot(
            snapshot_id=_snapshot_id(run_id, round_index, "survivor", survivor_refs),
            round_index=round_index,
            kind="survivor",
            candidate_refs=survivor_refs,
            merge_audit=merge.audit,
            run_id=run_id,
            workflow_ids=_workflow_ids(
                [
                    *_snapshot_workflow_ids(previous_snapshot, self.records_by_version),
                    *_snapshot_workflow_ids(current_snapshot, self.records_by_version),
                    *(record.workflow_id for record in source_records),
                ]
            ),
        )
        self._save_snapshot(run_id, snapshot)
        return snapshot

    def filtered_survivor_snapshot(
        self,
        *,
        run_id: str,
        round_index: int,
        source_snapshot_id: str | None,
        selected_candidate_ids: list[str],
        reason: str,
    ) -> CandidateSnapshot:
        source_snapshot = self._snapshot(source_snapshot_id)
        source_refs = list(source_snapshot.candidate_refs) if source_snapshot else []
        selected_ids = _ordered_unique_candidate_ids(selected_candidate_ids)
        selected_id_set = set(selected_ids)
        selected_refs = [
            CandidateSnapshotRef(candidate_id=ref.candidate_id, version_id=ref.version_id)
            for ref in source_refs
            if ref.candidate_id in selected_id_set
        ]
        source_candidate_ids = [ref.candidate_id for ref in source_refs]
        source_candidate_id_set = set(source_candidate_ids)
        ignored_ids = [
            candidate_id
            for candidate_id in selected_ids
            if candidate_id not in source_candidate_id_set
        ]
        snapshot = CandidateSnapshot(
            snapshot_id=_snapshot_id(run_id, round_index, "survivor", selected_refs),
            round_index=round_index,
            kind="survivor",
            candidate_refs=selected_refs,
            merge_audit=_filtered_survivor_merge_audit(
                source_snapshot=source_snapshot,
                source_candidate_ids=source_candidate_ids,
                requested_candidate_ids=list(selected_candidate_ids),
                selected_candidate_ids=selected_ids,
                survivor_refs=selected_refs,
                unknown_candidate_ids=ignored_ids,
                reason=reason,
            ),
            run_id=run_id,
            workflow_ids=_snapshot_workflow_ids(source_snapshot, self.records_by_version),
        )
        self._save_snapshot(run_id, snapshot)
        return snapshot

    def materialize(self, snapshot_id: str | None) -> list[dict[str, Any]]:
        if not snapshot_id:
            return []
        snapshot = self._snapshot(snapshot_id)
        if snapshot is None:
            return []
        materialized: list[dict[str, Any]] = []
        # Snapshot refs are immutable. Executor materialization must never
        # hydrate "the latest" record for a candidate id.
        for ref in snapshot.candidate_refs:
            record = self._record_by_version(ref.version_id)
            if record is None or record.status == "rejected":
                continue
            candidate = copy.deepcopy(record.candidate)
            validate_candidate_v1(candidate)
            materialized.append(candidate)
        return materialized

    def materialize_executor_outputs(
        self,
        *,
        snapshot_id: str | None,
        round_number: int,
        reason: str,
        summary: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            "outputs": {
                "prior_round": {
                    "round_number": round_number,
                    "candidates": self.materialize(snapshot_id),
                    "reason": reason,
                    "summary": summary or {},
                }
            }
        }

    def candidate_summary(self, snapshot_id: str | None, *, limit: int = 10) -> dict[str, Any]:
        return _candidate_summary(self.materialize(snapshot_id), limit=limit)

    def failure_cards(self, snapshot_id: str | None = None) -> list[dict[str, Any]]:
        snapshot = self._snapshot(snapshot_id)
        scoped_round = snapshot.round_index if snapshot is not None else None
        scoped_run_id = snapshot.run_id if snapshot is not None else None
        scoped_workflow_ids = set(snapshot.workflow_ids) if snapshot is not None else set()
        cards: list[dict[str, Any]] = []
        for record in self._candidate_records():
            if record.status != "rejected":
                continue
            if scoped_round is not None and record.round_index != scoped_round:
                continue
            if scoped_run_id and record.run_id != scoped_run_id:
                continue
            if scoped_workflow_ids and not scoped_workflow_ids.intersection(
                _record_workflow_ids(record)
            ):
                continue
            for error in record.validation_errors:
                cards.append(
                    {
                        "round_index": record.round_index,
                        "node_id": record.node_id,
                        "tool": _raw_source_tool(record.raw_source),
                        "phase": "candidate_archive_ingest",
                        "error_kind": error.get("error_kind", "candidate_validation_error"),
                        "field": error.get("field"),
                        "message": error.get("message"),
                        "repair_hint": error.get("repair_hint"),
                        "retryable": False,
                    }
                )
        return cards

    def _candidate_records(self) -> list[CandidateRecord]:
        records = dict(self.records_by_version)
        if self.session_store is not None and hasattr(self.session_store, "list_candidate_records"):
            for row in self.session_store.list_candidate_records():
                if isinstance(row, dict):
                    record = CandidateRecord(**row)
                    records.setdefault(record.version_id, record)
        return list(records.values())

    def capability_summary(self, snapshot_id: str | None) -> dict[str, Any]:
        capabilities: dict[str, int] = {}
        for candidate in self.materialize(snapshot_id):
            for capability in _candidate_capabilities(candidate):
                capabilities[capability] = capabilities.get(capability, 0) + 1
        return {"capabilities": capabilities, "count": sum(capabilities.values())}

    def snapshot_merge_summary(self, snapshot_id: str | None) -> dict[str, int]:
        snapshot = self._snapshot(snapshot_id)
        return summarize_candidate_merge_audit(snapshot.merge_audit if snapshot else {})

    def _raw_workflow_summary(self, workflow_id: str | None) -> dict[str, Any]:
        return self.raw_workflow_summary(workflow_id)

    def raw_workflow_summary(self, workflow_id: str | None) -> dict[str, Any]:
        if not workflow_id:
            return {"workflow_id": None, "nodes": []}
        try:
            wf = WorkflowStore(self.workspace).get(workflow_id)
            nodes = WorkflowStore(self.workspace).list_nodes(workflow_id)
        except Exception:
            return {"workflow_id": workflow_id, "nodes": []}
        return {
            "workflow_id": workflow_id,
            "state": wf.state,
            "nodes": [
                {
                    "node_id": node.node_id,
                    "tool": node.tool_name,
                    "job_id": node.job_id,
                    "state": node.state,
                    "output": copy.deepcopy(node.output),
                    "error_code": node.error_code,
                    "error_message": node.error_message,
                    "error_details": copy.deepcopy(node.error_details),
                }
                for node in nodes
            ],
        }

    def _record(
        self,
        *,
        run_id: str,
        round_index: int,
        workflow_id: str | None,
        status: Literal["current", "survivor", "rejected"],
        candidate: dict[str, Any],
        raw_source: dict[str, Any],
        validation_errors: list[dict[str, Any]],
    ) -> CandidateRecord:
        candidate_id = str(candidate.get("candidate_id") or f"rejected:{_digest(raw_source)}")
        version_payload = {
            "run_id": run_id,
            "round": round_index,
            "status": status,
            "candidate": candidate,
            "source": raw_source,
        }
        return CandidateRecord(
            run_id=run_id,
            candidate_id=candidate_id,
            version_id=f"cver_{_digest(version_payload)[:16]}",
            round_index=round_index,
            workflow_id=workflow_id,
            node_id=str(raw_source.get("node_id") or "") or None,
            job_id=str(raw_source.get("job_id") or "") or None,
            status=status,
            candidate=copy.deepcopy(candidate),
            raw_source=copy.deepcopy(raw_source),
            validation_errors=copy.deepcopy(validation_errors),
        )

    def _save_record(self, run_id: str, record: CandidateRecord) -> None:
        self.records_by_version[record.version_id] = record
        if self.session_store is not None and hasattr(self.session_store, "add_candidate_record"):
            self.session_store.add_candidate_record(run_id, asdict(record))

    def _save_snapshot(self, run_id: str, snapshot: CandidateSnapshot) -> None:
        self.snapshots[snapshot.snapshot_id] = snapshot
        if self.session_store is not None and hasattr(self.session_store, "add_candidate_snapshot"):
            payload = asdict(snapshot)
            payload["candidate_ids"] = snapshot.candidate_ids
            self.session_store.add_candidate_snapshot(run_id, payload)

    def _record_by_version(self, version_id: str) -> CandidateRecord | None:
        record = self.records_by_version.get(version_id)
        if record is not None:
            return record
        if self.session_store is not None and hasattr(
            self.session_store,
            "get_candidate_record_by_version",
        ):
            row = self.session_store.get_candidate_record_by_version(version_id)
            if isinstance(row, dict):
                record = CandidateRecord(**row)
                self.records_by_version[record.version_id] = record
                return record
        return None

    def _records_from_snapshot(self, snapshot: CandidateSnapshot | None) -> list[CandidateRecord]:
        if snapshot is None:
            return []
        records: list[CandidateRecord] = []
        for ref in snapshot.candidate_refs:
            record = self._record_by_version(ref.version_id)
            if record is not None:
                records.append(record)
        return records

    def _snapshot(self, snapshot_id: str | None) -> CandidateSnapshot | None:
        if not snapshot_id:
            return None
        snapshot = self.snapshots.get(snapshot_id)
        if snapshot is not None:
            return snapshot
        return self._load_snapshot(snapshot_id)

    def _load_snapshot(self, snapshot_id: str) -> CandidateSnapshot | None:
        if self.session_store is None or not hasattr(self.session_store, "get_candidate_snapshot"):
            return None
        row = self.session_store.get_candidate_snapshot(snapshot_id)
        if not isinstance(row, dict):
            return None
        refs = row.get("candidate_refs")
        if not isinstance(refs, list):
            refs = []
        row = {
            **row,
            "candidate_refs": [
                ref
                if isinstance(ref, CandidateSnapshotRef)
                else CandidateSnapshotRef(**ref)
                for ref in refs
                if isinstance(ref, (dict, CandidateSnapshotRef))
            ],
        }
        row.pop("candidate_ids", None)
        snapshot = CandidateSnapshot(**row)
        self.snapshots[snapshot.snapshot_id] = snapshot
        return snapshot


def _raw_candidate_items(summary: dict[str, Any]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    workflow_id = str(summary.get("workflow_id") or "") or None
    for node in summary.get("nodes") or []:
        if not isinstance(node, dict):
            continue
        output = node.get("output")
        if not isinstance(output, dict):
            continue
        node_id = str(node.get("node_id") or "")
        job_id = str(node.get("job_id") or "")
        tool = str(node.get("tool") or "")
        for key in _candidate_container_keys(output):
            values = output.get(key)
            if not isinstance(values, list):
                continue
            for index, raw in enumerate(values):
                candidate, errors = _canonicalize_candidate(
                    raw,
                    key=key,
                    tool=tool,
                    job_id=job_id,
                    node_id=node_id,
                    index=index,
                )
                items.append(
                    {
                        "candidate": candidate,
                        "candidate_errors": errors,
                        "raw_item": copy.deepcopy(raw),
                        "node_id": node_id,
                        "job_id": job_id,
                        "workflow_id": workflow_id,
                        "tool": tool,
                        "container_key": key,
                        "index": index,
                    }
                )
    return items


def _candidate_container_keys(output: dict[str, Any]) -> tuple[str, ...]:
    canonical = tuple(
        key
        for key in _CANONICAL_CANDIDATE_CONTAINERS
        if isinstance(output.get(key), list) and output.get(key)
    )
    if canonical:
        return canonical
    return tuple(
        key
        for key in _FALLBACK_CANDIDATE_CONTAINERS
        if isinstance(output.get(key), list) and output.get(key)
    )


def _canonicalize_candidate(
    item: Any,
    *,
    key: str,
    tool: str,
    job_id: str,
    node_id: str,
    index: int,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    if not isinstance(item, dict):
        return None, [_validation_error(None, "candidate", "candidate row is not an object")]
    structural_errors = _compact_field_errors(item)
    if structural_errors:
        return copy.deepcopy(item), structural_errors
    if _is_candidate_v1(item):
        return copy.deepcopy(item), []
    capabilities = _candidate_capabilities(item)
    if not capabilities.intersection(_MATERIAL_CAPABILITIES):
        return None, [
            _validation_error(
                None,
                "capabilities",
                "candidate row has no material candidate capability",
            )
        ]
    explicit_id = item.get("candidate_id")
    local_id = str(explicit_id or item.get("id") or f"{key}:{index}")
    source_tool = tool or str(item.get("tool") or key)
    candidate_id = (
        str(explicit_id).strip()
        if explicit_id is not None and str(explicit_id).strip()
        else f"{source_tool}:{job_id or node_id or 'unknown'}:{local_id}"
    )
    candidate: dict[str, Any] = {
        "schema_version": "candidate_v1",
        "candidate_id": candidate_id,
        "source": {
            "tool": source_tool,
            "node_id": node_id or None,
            "job_id": job_id or None,
            "local_id": local_id,
            "index": index,
        },
        "capabilities": {
            cap: True
            for cap in sorted(capabilities.intersection(_MATERIAL_CAPABILITIES))
        },
    }
    for field in ("sequence", "structure_path", "backbone_path", "full_structure_path"):
        if isinstance(item.get(field), str) and item.get(field).strip():
            candidate[field] = item[field]
    for field in ("scores", "artifacts"):
        value = item.get(field)
        if isinstance(value, dict):
            candidate[field] = copy.deepcopy(value)
    return candidate, []


def _candidate_validation_errors(candidate: Any) -> list[dict[str, Any]]:
    if not isinstance(candidate, dict):
        return [_validation_error(None, "candidate", "candidate is not an object")]
    errors: list[dict[str, Any]] = []
    errors.extend(_compact_field_errors(candidate))
    try:
        validate_candidate_v1(candidate)
    except ValidationError as exc:
        errors.append(
            _validation_error(
                "candidate_schema_error",
                ".".join(str(part) for part in exc.absolute_path) or None,
                exc.message,
            )
        )
    return errors


def _compact_field_errors(item: dict[str, Any]) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    for field in ("scores", "artifacts"):
        value = item.get(field)
        if value is None or isinstance(value, dict):
            continue
        errors.append(
            _validation_error(
                "compact_candidate_field",
                field,
                f"{field} must be an object for executor materialization",
                repair_hint=(
                    "Read the raw workflow output or candidate archive, "
                    "not a compact report summary."
                ),
            )
        )
    return errors


def _validation_error(
    error_kind: str | None,
    field: str | None,
    message: str,
    *,
    repair_hint: str | None = None,
) -> dict[str, Any]:
    return {
        "error_kind": error_kind or "candidate_validation_error",
        "field": field,
        "message": message,
        "repair_hint": repair_hint,
    }


def _candidate_capabilities(candidate: dict[str, Any]) -> set[str]:
    capabilities: set[str] = set()
    raw = candidate.get("capabilities")
    explicitly_false: set[str] = set()
    if isinstance(raw, dict):
        capabilities.update(
            str(name)
            for name, value in raw.items()
            if str(name).startswith("has_") and value is True
        )
        explicitly_false.update(
            str(name)
            for name, value in raw.items()
            if str(name).startswith("has_") and value is False
        )
    elif isinstance(raw, list):
        capabilities.update(str(item) for item in raw if isinstance(item, str))
    if (
        "has_sequence" not in explicitly_false
        and isinstance(candidate.get("sequence"), str)
        and candidate["sequence"].strip()
    ):
        capabilities.add("has_sequence")
    if (
        "has_backbone_structure" not in explicitly_false
        and (candidate.get("structure_path") or candidate.get("backbone_path"))
    ):
        capabilities.add("has_backbone_structure")
    if (
        "has_full_structure" not in explicitly_false
        and (candidate.get("full_structure_path") or candidate.get("structure_path"))
    ):
        capabilities.add("has_full_structure")
    scores = candidate.get("scores")
    if "has_candidate_scores" not in explicitly_false and isinstance(scores, dict) and scores:
        capabilities.add("has_candidate_scores")
    return capabilities


def _candidate_summary(candidates: list[dict[str, Any]], *, limit: int = 10) -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    for candidate in candidates[:limit]:
        source = candidate.get("source") if isinstance(candidate.get("source"), dict) else {}
        capabilities = candidate.get("capabilities")
        scores = candidate.get("scores")
        items.append(
            {
                "candidate_id": candidate.get("candidate_id"),
                "source": {
                    "tool": source.get("tool"),
                    "node_id": source.get("node_id"),
                    "job_id": source.get("job_id"),
                },
                "capabilities": dict(capabilities) if isinstance(capabilities, dict) else {},
                "has_sequence": bool(candidate.get("sequence")),
                "has_structure": bool(
                    candidate.get("structure_path")
                    or candidate.get("backbone_path")
                    or candidate.get("full_structure_path")
                ),
                "score_keys": sorted(scores) if isinstance(scores, dict) else [],
            }
        )
    return {"count": len(candidates), "items": items, "truncated": len(candidates) > limit}


def _is_candidate_v1(candidate: Any) -> bool:
    return isinstance(candidate, dict) and candidate.get("schema_version") == "candidate_v1"


def _snapshot_id(
    run_id: str,
    round_index: int,
    kind: str,
    refs: list[CandidateSnapshotRef],
) -> str:
    payload = {
        "run_id": run_id,
        "round": round_index,
        "kind": kind,
        "refs": [asdict(ref) for ref in refs],
    }
    return f"csnap_{_digest(payload)[:16]}"


def _ordered_unique_candidate_ids(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value is None:
            continue
        text = str(value)
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _filtered_survivor_merge_audit(
    *,
    source_snapshot: CandidateSnapshot | None,
    source_candidate_ids: list[str],
    requested_candidate_ids: list[str],
    selected_candidate_ids: list[str],
    survivor_refs: list[CandidateSnapshotRef],
    unknown_candidate_ids: list[str],
    reason: str,
) -> dict[str, Any]:
    retained_candidate_ids = [ref.candidate_id for ref in survivor_refs]
    return {
        "source": "filtered_survivor_snapshot",
        "source_snapshot_id": source_snapshot.snapshot_id if source_snapshot else None,
        "reason": reason,
        "requested_candidate_ids": requested_candidate_ids,
        "selected_candidate_ids": retained_candidate_ids,
        "ignored_candidate_ids": unknown_candidate_ids,
        "input_count": len(source_candidate_ids),
        "output_count": len(retained_candidate_ids),
        "duplicates_removed": 0,
        "duplicate_groups": [],
        "conflicts": [],
        "operation": "filtered_survivor_snapshot",
        "filter_audit": {
            "source_snapshot_id": source_snapshot.snapshot_id if source_snapshot else None,
            "source_snapshot_kind": source_snapshot.kind if source_snapshot else None,
            "source_candidate_ids": source_candidate_ids,
            "selected_candidate_ids": selected_candidate_ids,
            "retained_candidate_ids": retained_candidate_ids,
            "unknown_candidate_ids": unknown_candidate_ids,
            "ignored_unknown_candidate_ids": unknown_candidate_ids,
        },
    }


def _digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _workflow_ids(values: list[Any]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    pending = list(values)
    while pending:
        value = pending.pop(0)
        if isinstance(value, (list, tuple, set)):
            pending[0:0] = list(value)
            continue
        if value in (None, ""):
            continue
        text = str(value)
        if text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _snapshot_workflow_ids(
    snapshot: CandidateSnapshot | None,
    records_by_version: dict[str, CandidateRecord],
) -> list[str]:
    if snapshot is None:
        return []
    if snapshot.workflow_ids:
        return _workflow_ids(list(snapshot.workflow_ids))
    return _workflow_ids(
        [
            records_by_version.get(ref.version_id).workflow_id
            for ref in snapshot.candidate_refs
            if records_by_version.get(ref.version_id) is not None
        ]
    )


def _record_workflow_ids(record: CandidateRecord) -> set[str]:
    values: list[Any] = [record.workflow_id]
    raw_source = record.raw_source if isinstance(record.raw_source, dict) else {}
    values.extend(
        [
            raw_source.get("workflow_id"),
            raw_source.get("source_workflow_id"),
            raw_source.get("representative_workflow_id"),
            raw_source.get("source_workflow_ids"),
        ]
    )
    candidate_source = record.candidate.get("source")
    if isinstance(candidate_source, dict):
        values.extend(
            [
                candidate_source.get("workflow_id"),
                candidate_source.get("source_workflow_id"),
            ]
        )
    return set(_workflow_ids(values))


def _candidate_merge_positions(candidate: dict[str, Any]) -> list[int]:
    positions = candidate.get("__candidate_merge_positions")
    if not isinstance(positions, list):
        return []
    out: list[int] = []
    for item in positions:
        try:
            out.append(int(item))
        except (TypeError, ValueError):
            continue
    return out


def _candidate_merge_representative_position(candidate: dict[str, Any]) -> int | None:
    try:
        return int(candidate.get("__candidate_merge_representative_position"))
    except (TypeError, ValueError):
        return None


def _strip_candidate_merge_private(candidate: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in candidate.items() if not str(key).startswith("__")}


def _source_records_for_merged_candidate(
    candidate: dict[str, Any],
    source_records: list[CandidateRecord],
) -> list[CandidateRecord]:
    positions = _candidate_merge_positions(candidate)
    if not positions:
        representative_position = _candidate_merge_representative_position(candidate)
        positions = [representative_position] if representative_position is not None else []
    out: list[CandidateRecord] = []
    for position in positions:
        if 0 <= position < len(source_records):
            out.append(source_records[position])
    return out


def _representative_record_for_merged_candidate(
    candidate: dict[str, Any],
    source_records: list[CandidateRecord],
) -> CandidateRecord | None:
    representative_position = _candidate_merge_representative_position(candidate)
    if representative_position is not None and 0 <= representative_position < len(source_records):
        return source_records[representative_position]
    source_lineage = _source_records_for_merged_candidate(candidate, source_records)
    return source_lineage[0] if source_lineage else None


def _survivor_raw_source(
    *,
    previous_snapshot: CandidateSnapshot | None,
    current_snapshot: CandidateSnapshot | None,
    representative_record: CandidateRecord | None,
    source_records: list[CandidateRecord],
) -> dict[str, Any]:
    source_workflow_ids = _workflow_ids([record.workflow_id for record in source_records])
    raw_source: dict[str, Any] = {
        "source": "survivor_merge",
        "previous_snapshot_id": previous_snapshot.snapshot_id if previous_snapshot else None,
        "current_snapshot_id": current_snapshot.snapshot_id if current_snapshot else None,
        "source_version_ids": [record.version_id for record in source_records],
        "source_candidate_ids": [record.candidate_id for record in source_records],
        "source_workflow_ids": source_workflow_ids,
    }
    if representative_record is not None:
        raw_source.update(
            {
                "representative_version_id": representative_record.version_id,
                "representative_candidate_id": representative_record.candidate_id,
                "representative_workflow_id": representative_record.workflow_id,
                "workflow_id": representative_record.workflow_id,
                "node_id": representative_record.node_id,
                "job_id": representative_record.job_id,
                "tool": _raw_source_tool(representative_record.raw_source),
            }
        )
    return {key: value for key, value in raw_source.items() if value not in (None, [], {})}


def _raw_source_tool(raw_source: dict[str, Any]) -> str | None:
    value = raw_source.get("tool")
    return str(value) if value else None


__all__ = [
    "CandidateArchive",
    "CandidateRecord",
    "CandidateSnapshot",
    "CandidateSnapshotRef",
]
