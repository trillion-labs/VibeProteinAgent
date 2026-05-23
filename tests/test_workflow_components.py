from __future__ import annotations

from vibeproteinagent.workflows.descriptor import WorkflowNode
from vibeproteinagent.workflows.failure import FailurePolicy
from vibeproteinagent.workflows.planner import WorkflowPlanner
from vibeproteinagent.workflows.state import NodeRuntime, WorkflowStateMachine


def test_planner_static_upstream_handles_foreach_children():
    node = WorkflowNode(
        id="score",
        tool="noop",
        inputs={"x": "${outputs.designs[0].pdb_path}"},
        foreach="${outputs.designs.items}",
    )
    assert WorkflowPlanner.static_upstream(node) == ["designs"]


def test_planner_static_upstream_handles_job_refs():
    node = WorkflowNode(
        id="submit",
        tool="submit_candidate",
        inputs={"candidates": [{"job_id": "${jobs.rank}"}]},
    )
    assert WorkflowPlanner.static_upstream(node) == ["rank"]


def test_state_machine_builds_ordered_foreach_scope():
    base = WorkflowNode(id="n", tool="noop", inputs={})
    first = NodeRuntime(
        desc=base,
        runtime_id="n[0]",
        state="completed",
        output={"x": 1},
        foreach_parent="n",
        foreach_index=0,
    )
    second = NodeRuntime(
        desc=base,
        runtime_id="n[1]",
        state="completed",
        output={"x": 2},
        foreach_parent="n",
        foreach_index=1,
    )
    scope = WorkflowStateMachine.build_scope({}, {"n[0]": first, "n[1]": second})
    assert scope["outputs"]["n"] == [{"x": 1}, {"x": 2}]


def test_state_machine_exposes_job_refs():
    node = WorkflowNode(id="rank", tool="rank_designs", inputs={})
    runtime = NodeRuntime(
        desc=node,
        runtime_id="rank",
        state="completed",
        job_id="job_rank",
        output={"ranked_candidates": []},
    )
    scope = WorkflowStateMachine.build_scope({}, {"rank": runtime})
    assert scope["jobs"]["rank"] == "job_rank"


def test_state_machine_prefers_completed_foreach_parent_aggregate():
    base_node = WorkflowNode(id="n", tool="noop", inputs={}, foreach="${params.items}")
    child_node = WorkflowNode(id="n", tool="noop", inputs={})
    parent = NodeRuntime(
        desc=base_node,
        runtime_id="n",
        state="completed",
        output=[{"x": 1}, {"x": 2}],
    )
    first = NodeRuntime(
        desc=child_node,
        runtime_id="n[0]",
        state="completed",
        output={"x": 1},
        foreach_parent="n",
        foreach_index=0,
    )
    second = NodeRuntime(
        desc=child_node,
        runtime_id="n[1]",
        state="completed",
        output={"x": 2},
        foreach_parent="n",
        foreach_index=1,
    )

    scope = WorkflowStateMachine.build_scope(
        {},
        {"n": parent, "n[0]": first, "n[1]": second},
    )

    assert scope["outputs"]["n"] == [{"x": 1}, {"x": 2}]
    assert scope["outputs"]["n[0]"] == {"x": 1}
    assert scope["outputs"]["n[1]"] == {"x": 2}


def test_state_machine_keeps_foreach_parent_job_refs_after_aggregate():
    base_node = WorkflowNode(id="n", tool="noop", inputs={}, foreach="${params.items}")
    child_node = WorkflowNode(id="n", tool="noop", inputs={})
    parent = NodeRuntime(
        desc=base_node,
        runtime_id="n",
        state="completed",
        output=[{"x": 1}, {"x": 2}],
    )
    first = NodeRuntime(
        desc=child_node,
        runtime_id="n[0]",
        state="completed",
        job_id="job_first",
        output={"x": 1},
        foreach_parent="n",
        foreach_index=0,
    )
    second = NodeRuntime(
        desc=child_node,
        runtime_id="n[1]",
        state="completed",
        job_id="job_second",
        output={"x": 2},
        foreach_parent="n",
        foreach_index=1,
    )

    scope = WorkflowStateMachine.build_scope(
        {},
        {"n": parent, "n[0]": first, "n[1]": second},
    )

    assert scope["jobs"]["n"] == ["job_first", "job_second"]


def test_failure_policy_finds_transitive_descendants():
    a = NodeRuntime(desc=WorkflowNode(id="a", tool="noop", inputs={}), runtime_id="a")
    b = NodeRuntime(
        desc=WorkflowNode(id="b", tool="noop", inputs={}),
        runtime_id="b",
        upstream=["a"],
    )
    c = NodeRuntime(
        desc=WorkflowNode(id="c", tool="noop", inputs={}),
        runtime_id="c",
        upstream=["b"],
    )
    assert FailurePolicy.descendant_runtime_ids(a, {"a": a, "b": b, "c": c}) == {
        "b",
        "c",
    }
