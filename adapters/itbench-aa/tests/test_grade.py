#!/usr/bin/env python3
"""
Regression and unit tests for the itbench-aa grader
(src/itbench_aa/task-template/tests/grade.py).

Covers:
- New v1.0 answer schema (root_causes array, schema_version)
- Recall-gated precision headline metric
- Per-entity precision: correct + N spurious -> 1/(N+1)
- Recall gate: no GT group matched -> 0.0
- Duplicate submissions: second claim on the same GT group lands in denominator
- Aliased-sibling matching: Service where GT is Pod, resolved via alias class
- Multi-root-cause scenarios (fictitious GT): partial match fires recall gate
- Transcript recovery: reward always 0.0 (no partial credit on new schema)
- Namespace matching (applicable / not applicable)
- Chain metrics: chain_applicable, chain_head_correct, propagation_edge_coverage
- Reasoning presence via token resolution (not char count)
- Format validity gates

Runnable standalone:
    python3 adapters/itbench-aa/tests/test_grade.py
Also pytest-discoverable (test_-prefixed functions).
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import sys
import tempfile
from pathlib import Path

GRADE_PY = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "itbench_aa"
    / "task-template"
    / "tests"
    / "grade.py"
)

# ---------------------------------------------------------------------------
# Ground-truth fixtures
# ---------------------------------------------------------------------------

MINIMAL_GT = r"""
groups:
  - id: root-pod-1
    kind: Pod
    filter:
      - root-pod-.*
    root_cause: true
  - id: other-service-1
    kind: Service
    filter:
      - other-service\b
propagations:
  - source: root-pod-1
    target: other-service-1
    condition: root pod misbehaves
    effect: other service errors
"""

# GT with namespace on the root cause group
NAMESPACE_GT = r"""
groups:
  - id: root-pod-1
    kind: Pod
    namespace: otel-demo
    filter:
      - root-pod-.*
    root_cause: true
"""

# GT with no propagations (scenarios 37, 38 style)
NO_PROPAGATIONS_GT = r"""
groups:
  - id: root-pod-1
    kind: Pod
    filter:
      - root-pod-.*
    root_cause: true
"""

# GT with a Deployment using pod-suffix filter -- tests widening patch class
DEPLOYMENT_SUFFIX_GT = r"""
groups:
  - id: myapp-deployment
    kind: Deployment
    filter:
      - myapp(-.*)?
    root_cause: true
"""

# GT with alias class: Pod (root cause) and Service (alias sibling)
ALIAS_GT = r"""
groups:
  - id: root-pod-1
    kind: Pod
    namespace: otel-demo
    filter:
      - root-pod-.*
    root_cause: true
  - id: root-svc-1
    kind: Service
    namespace: otel-demo
    filter:
      - root-svc\b
aliases:
  - - root-pod-1
    - root-svc-1
propagations:
  - source: root-pod-1
    target: root-svc-1
    condition: pod crashes
    effect: service down
"""

# Scenario-1-shaped GT: a workload-config fault modelled as a Pod root cause,
# with the workload's own Service aliased in, PLUS a downstream victim workload
# (frontend-proxy Pod+Service) that the ground-truth author also placed in the
# root's alias class. Exercises: (a) controller<->Pod<->Service workload
# equivalence claiming the Pod root by ROOT identity, (b) the guardrail that a
# foreign/victim workload cannot reach the root through the new equivalence path,
# and (c) the fact that we RESPECT the authored alias block (a differently-named
# sibling the author listed is an accepted answer -- we do not override the
# benchmark's own ground truth).
WORKLOAD_EQUIV_GT = r"""
groups:
  - id: load-generator-pod-1
    kind: Pod
    filter:
      - load-generator-.*
    namespace: otel-demo
    root_cause: true
  - id: load-generator-service-1
    kind: Service
    filter:
      - load-generator\b
    namespace: otel-demo
  - id: frontend-proxy-service-1
    kind: Service
    filter:
      - frontend-proxy\b
    namespace: otel-demo
  - id: frontend-proxy-pod-1
    kind: Pod
    filter:
      - frontend-proxy-.*
    namespace: otel-demo
aliases:
  - - load-generator-service-1
    - load-generator-pod-1
    - frontend-proxy-service-1
    - frontend-proxy-pod-1
propagations:
  - source: load-generator-pod-1
    target: load-generator-service-1
    condition: too many users
    effect: high request volume
  - source: load-generator-service-1
    target: frontend-proxy-service-1
    condition: overloaded
    effect: error rate above threshold
"""

# Fictitious two-root-cause GT to exercise the recall gate
MULTI_RC_GT = r"""
groups:
  - id: pod-a
    kind: Pod
    filter:
      - pod-a-.*
    root_cause: true
  - id: pod-b
    kind: Pod
    filter:
      - pod-b-.*
    root_cause: true
"""


# ---------------------------------------------------------------------------
# Test harness helpers
# ---------------------------------------------------------------------------

def _load_grade(tmp_path: Path, ground_truth_yaml: str = MINIMAL_GT):
    """Import a fresh grade module instance with path constants monkeypatched."""
    grade_dir = str(GRADE_PY.parent)
    sys.path.insert(0, grade_dir)
    try:
        spec = importlib.util.spec_from_file_location(f"grade_{id(tmp_path)}", GRADE_PY)
        grade = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(grade)
    finally:
        sys.path.remove(grade_dir)

    workspace = tmp_path / "workspace"
    logs_verifier = tmp_path / "logs" / "verifier"
    logs_agent = tmp_path / "logs" / "agent"
    workspace.mkdir(parents=True)
    logs_agent.mkdir(parents=True)
    (tmp_path / "ground_truth.yaml").write_text(ground_truth_yaml)

    grade.ANSWER_PATH = workspace / "answer.json"
    grade.GROUND_TRUTH_PATH = tmp_path / "ground_truth.yaml"
    grade.REWARD_DIR = logs_verifier
    grade.REWARD_PATH = logs_verifier / "reward.json"
    grade.AGENT_LOGS_DIR = logs_agent
    return grade


def _run(grade) -> dict:
    grade.main()
    return json.loads(grade.REWARD_PATH.read_text())


def _run_capturing_stdout(grade) -> tuple[dict, str]:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        grade.main()
    return json.loads(grade.REWARD_PATH.read_text()), buf.getvalue()


def _write_answer(grade, root_causes: list[dict], **extra) -> None:
    """Write a v1.0 schema answer.json."""
    answer = {"schema_version": "1.0", "root_causes": root_causes, **extra}
    grade.ANSWER_PATH.write_text(json.dumps(answer))


def _write_transcript_new_schema(grade, root_causes: list[dict]) -> None:
    """Simulate an agent that stated its answer in chat (new schema) but never
    wrote answer.json."""
    transcript = grade.AGENT_LOGS_DIR / "session.jsonl"
    blob = json.dumps({"schema_version": "1.0", "root_causes": root_causes})
    entry = {
        "message": {
            "role": "assistant",
            "content": [{"type": "text", "text": f"My conclusion: {blob}"}],
        }
    }
    transcript.write_text(json.dumps(entry) + "\n")


def _write_transcript_legacy(grade, root_cause: str, kind: str) -> None:
    """Simulate an agent that stated its answer in the old flat schema."""
    transcript = grade.AGENT_LOGS_DIR / "session.jsonl"
    entry = {
        "message": {
            "role": "assistant",
            "content": [{"type": "text", "text": f'My answer: {{"root_cause": "{root_cause}", "kind": "{kind}"}}'}],
        }
    }
    transcript.write_text(json.dumps(entry) + "\n")


# ---------------------------------------------------------------------------
# Precision / recall tests  (the untested headline metric)
# ---------------------------------------------------------------------------

def test_single_correct_entity_full_reward():
    """One correct entity, nothing else -> reward=1.0."""
    with tempfile.TemporaryDirectory() as td:
        grade = _load_grade(Path(td))
        _write_answer(grade, [{"name": "root-pod-xyz", "kind": "Pod"}],
                      reasoning="root-pod-1 crashed",
                      propagation_chain=["root-pod-1", "other-service"])
        result = _run(grade)
        assert result["reward"] == 1.0, result
        assert result["name_match"] == 1.0, result
        assert result["kind_match"] == 1.0, result
        assert result["submitted_entity_count"] == 1.0, result


def test_correct_plus_one_spurious_halves_reward():
    """Correct entity + 1 spurious -> precision = 1/2 -> reward=0.5."""
    with tempfile.TemporaryDirectory() as td:
        grade = _load_grade(Path(td))
        _write_answer(grade, [
            {"name": "root-pod-xyz", "kind": "Pod"},
            {"name": "totally-unrelated", "kind": "Pod"},
        ], reasoning="root-pod-1 crashed")
        result = _run(grade)
        assert result["reward"] == 0.5, result
        assert result["submitted_entity_count"] == 2.0, result


def test_correct_plus_two_spurious_thirds_reward():
    """Correct entity + 2 spurious -> precision = 1/3."""
    with tempfile.TemporaryDirectory() as td:
        grade = _load_grade(Path(td))
        _write_answer(grade, [
            {"name": "root-pod-xyz", "kind": "Pod"},
            {"name": "unrelated-a", "kind": "Pod"},
            {"name": "unrelated-b", "kind": "Service"},
        ], reasoning="root-pod-1 crashed")
        result = _run(grade)
        assert abs(result["reward"] - 1/3) < 1e-9, result
        assert result["submitted_entity_count"] == 3.0, result


def test_wrong_entity_only_scores_zero():
    """No GT group matched -> recall gate fires -> reward=0.0."""
    with tempfile.TemporaryDirectory() as td:
        grade = _load_grade(Path(td))
        _write_answer(grade, [{"name": "totally-wrong", "kind": "Pod"}],
                      reasoning="totally-wrong crashed")
        result = _run(grade)
        assert result["reward"] == 0.0, result
        assert result["name_match"] == 0.0, result


def test_duplicate_submission_second_in_denominator():
    """Submitting the same entity twice: second claim of same GT group stays in
    denominator (matched_count stays 1, submitted_count=2) -> reward=0.5."""
    with tempfile.TemporaryDirectory() as td:
        grade = _load_grade(Path(td))
        _write_answer(grade, [
            {"name": "root-pod-abc", "kind": "Pod"},
            {"name": "root-pod-xyz", "kind": "Pod"},  # also matches root-pod-1
        ], reasoning="root-pod-1 crashed")
        result = _run(grade)
        # Both entities match the same GT group; second claim is rejected.
        # matched_count=1, submitted=2 -> precision=0.5
        assert result["reward"] == 0.5, result
        assert result["submitted_entity_count"] == 2.0, result


def test_multi_rc_one_of_two_correct_fires_recall_gate():
    """Fictitious two-root-cause GT: agent submits only pod-a.
    matched=1, gt_count=2 -> all_gt_covered=False -> recall gate -> reward=0.0."""
    with tempfile.TemporaryDirectory() as td:
        grade = _load_grade(Path(td), ground_truth_yaml=MULTI_RC_GT)
        _write_answer(grade, [{"name": "pod-a-xyz", "kind": "Pod"}],
                      reasoning="pod-a-xyz crashed")
        result = _run(grade)
        assert result["reward"] == 0.0, result


def test_multi_rc_both_correct_full_reward():
    """Both GT groups covered by one entity each -> reward=1.0."""
    with tempfile.TemporaryDirectory() as td:
        grade = _load_grade(Path(td), ground_truth_yaml=MULTI_RC_GT)
        _write_answer(grade, [
            {"name": "pod-a-xyz", "kind": "Pod"},
            {"name": "pod-b-xyz", "kind": "Pod"},
        ], reasoning="pod-a-xyz and pod-b-xyz crashed")
        result = _run(grade)
        assert result["reward"] == 1.0, result
        assert result["submitted_entity_count"] == 2.0, result


# ---------------------------------------------------------------------------
# Alias resolution
# ---------------------------------------------------------------------------

def test_alias_sibling_service_matches_pod_root_cause():
    """GT root is a Pod; agent submits kind=Service with the sibling's name ->
    resolves via alias class -> reward=1.0."""
    with tempfile.TemporaryDirectory() as td:
        grade = _load_grade(Path(td), ground_truth_yaml=ALIAS_GT)
        # Agent names the Service sibling, not the Pod root cause
        _write_answer(grade, [{"name": "root-svc", "kind": "Service",
                                "namespace": "otel-demo"}],
                      reasoning="root-pod-1 crashed")
        result = _run(grade)
        assert result["reward"] == 1.0, result
        assert result["name_match"] == 1.0, result
        assert result["kind_match"] == 1.0, result


# ---------------------------------------------------------------------------
# Workload equivalence (controller <-> Pod <-> Service) + guardrails
# ---------------------------------------------------------------------------

def test_workload_equiv_deployment_matches_pod_root():
    """The actual scenario-1 reward=0 bug: a workload-config fault is modelled as
    a Pod root cause, but naming the governing Deployment is the natural answer.
    {name: load-generator, kind: Deployment} must claim the Pod root via
    workload equivalence (gated on the ROOT's own suffix-relaxed pattern)."""
    with tempfile.TemporaryDirectory() as td:
        grade = _load_grade(Path(td), ground_truth_yaml=WORKLOAD_EQUIV_GT)
        _write_answer(grade, [{"name": "load-generator", "kind": "Deployment",
                                "namespace": "otel-demo"}],
                      reasoning="load-generator-pod-1 misconfigured LOCUST_USERS")
        result = _run(grade)
        assert result["reward"] == 1.0, result
        assert result["name_match"] == 1.0, result
        assert result["kind_match"] == 1.0, result


def test_workload_equiv_bare_pod_name_matches_root():
    """The bare workload name with kind=Pod (no hash suffix) must still claim the
    Pod root via the suffix-relaxed root identity."""
    with tempfile.TemporaryDirectory() as td:
        grade = _load_grade(Path(td), ground_truth_yaml=WORKLOAD_EQUIV_GT)
        _write_answer(grade, [{"name": "load-generator", "kind": "Pod",
                                "namespace": "otel-demo"}],
                      reasoning="load-generator-pod-1 misconfigured")
        result = _run(grade)
        assert result["reward"] == 1.0, result


def test_authored_alias_service_is_respected():
    """We do NOT override the benchmark's authored alias block. The ground-truth
    author placed frontend-proxy-service-1 in the root's alias class, so naming
    it (kind=Service, matching that sibling's own filter) is an accepted answer
    per ITBench-AA's own ground truth. Locking this in guards against a
    well-meaning 'victim gate' silently making the grader stricter than the
    benchmark declares."""
    with tempfile.TemporaryDirectory() as td:
        grade = _load_grade(Path(td), ground_truth_yaml=WORKLOAD_EQUIV_GT)
        _write_answer(grade, [{"name": "frontend-proxy", "kind": "Service",
                                "namespace": "otel-demo"}],
                      reasoning="frontend-proxy-service-1 in alias class")
        result = _run(grade)
        assert result["reward"] == 1.0, result


def test_workload_equiv_does_not_admit_victim_via_new_path():
    """Guardrail: the NEW workload-equivalence path must not create false
    positives. A victim workload's Deployment (frontend-proxy/Deployment) has no
    Deployment sibling to match and does not satisfy the ROOT's own pattern, so
    it must NOT claim the root through the new equivalence path -> reward=0.0."""
    with tempfile.TemporaryDirectory() as td:
        grade = _load_grade(Path(td), ground_truth_yaml=WORKLOAD_EQUIV_GT)
        _write_answer(grade, [{"name": "frontend-proxy", "kind": "Deployment",
                                "namespace": "otel-demo"}],
                      reasoning="frontend-proxy is the symptom")
        result = _run(grade)
        assert result["reward"] == 0.0, result
        assert result["name_match"] == 0.0, result


def test_workload_equiv_foreign_workload_does_not_match():
    """A completely unrelated workload's Deployment must never match the root via
    workload equivalence."""
    with tempfile.TemporaryDirectory() as td:
        grade = _load_grade(Path(td), ground_truth_yaml=WORKLOAD_EQUIV_GT)
        _write_answer(grade, [{"name": "adservice", "kind": "Deployment",
                                "namespace": "otel-demo"}],
                      reasoning="adservice unrelated")
        result = _run(grade)
        assert result["reward"] == 0.0, result
        assert result["name_match"] == 0.0, result


def test_chain_resolver_root_identity_resolves_to_root_head():
    """resolve_chain_element: the bare workload name (root identity) resolves to
    the ROOT group id -- so the chain head is correct even when a same-namespace
    Service shares the stem. A foreign/victim name must NOT resolve to the root
    head (it resolves to its own group instead)."""
    import yaml  # PyYAML is a declared dependency of the adapter
    with tempfile.TemporaryDirectory() as td:
        grade = _load_grade(Path(td), ground_truth_yaml=WORKLOAD_EQUIV_GT)
        gt = grade.normalize_ground_truth(yaml.safe_load(WORKLOAD_EQUIV_GT))
        groups_by_id = {g["id"]: g for g in gt["groups"]}
        root = grade.find_root_cause_group(gt)
        assert grade.resolve_chain_element("load-generator", groups_by_id, root) == "load-generator-pod-1"
        # frontend-proxy must not resolve to the root head
        assert grade.resolve_chain_element("frontend-proxy", groups_by_id, root) != "load-generator-pod-1"


# ---------------------------------------------------------------------------
# Deployment widening filter (patches applied at generation time)
# ---------------------------------------------------------------------------

def test_deployment_bare_name_matches_widened_filter():
    """Widened filter '(-.*)?': bare Deployment name without pod-hash suffix
    must pass STRICT fullmatch so a correct agent answer is not rejected."""
    with tempfile.TemporaryDirectory() as td:
        grade = _load_grade(Path(td), ground_truth_yaml=DEPLOYMENT_SUFFIX_GT)
        _write_answer(grade, [{"name": "myapp", "kind": "Deployment"}],
                      reasoning="myapp crashed")
        result = _run(grade)
        assert result["name_match"] == 1.0, result
        assert result["reward"] == 1.0, result


def test_deployment_pod_name_also_matches_widened_filter():
    """After widening, a pod name like myapp-7f9c-x2k must still match
    (widening must not narrow the pod case)."""
    with tempfile.TemporaryDirectory() as td:
        grade = _load_grade(Path(td), ground_truth_yaml=DEPLOYMENT_SUFFIX_GT)
        _write_answer(grade, [{"name": "myapp-7f9c-x2k", "kind": "Deployment"}],
                      reasoning="myapp crashed")
        result = _run(grade)
        assert result["name_match"] == 1.0, result
        assert result["reward"] == 1.0, result


def test_unrelated_name_does_not_match_widened_filter():
    """Widening must not introduce false positives."""
    with tempfile.TemporaryDirectory() as td:
        grade = _load_grade(Path(td), ground_truth_yaml=DEPLOYMENT_SUFFIX_GT)
        _write_answer(grade, [{"name": "totally-unrelated", "kind": "Deployment"}],
                      reasoning="crashed")
        result = _run(grade)
        assert result["name_match"] == 0.0, result
        assert result["reward"] == 0.0, result


# ---------------------------------------------------------------------------
# Format validity
# ---------------------------------------------------------------------------

def test_missing_root_causes_format_invalid():
    """Answer with no root_causes -> answer_format_valid=0.0."""
    with tempfile.TemporaryDirectory() as td:
        grade = _load_grade(Path(td))
        grade.ANSWER_PATH.write_text(json.dumps(
            {"schema_version": "1.0", "reasoning": "something"}
        ))
        result = _run(grade)
        assert result["answer_format_valid"] == 0.0, result
        assert result["reward"] == 0.0, result


def test_wrong_schema_version_format_invalid():
    """schema_version present but != '1.0' -> format_invalid."""
    with tempfile.TemporaryDirectory() as td:
        grade = _load_grade(Path(td))
        grade.ANSWER_PATH.write_text(json.dumps(
            {"schema_version": "2.0",
             "root_causes": [{"name": "root-pod-xyz", "kind": "Pod"}]}
        ))
        result = _run(grade)
        assert result["answer_format_valid"] == 0.0, result


def test_missing_schema_version_still_valid():
    """schema_version absent -> soft-HARD: log and proceed, format_ok=True."""
    with tempfile.TemporaryDirectory() as td:
        grade = _load_grade(Path(td))
        grade.ANSWER_PATH.write_text(json.dumps(
            {"root_causes": [{"name": "root-pod-xyz", "kind": "Pod"}],
             "reasoning": "root-pod-1 crashed",
             "propagation_chain": ["root-pod-1", "other-service"]}
        ))
        result = _run(grade)
        assert result["answer_format_valid"] == 1.0, result
        assert result["reward"] == 1.0, result


def test_legacy_flat_schema_on_disk_format_invalid():
    """Regression guard: instruction.md must never again tell the agent to
    write the pre-v1.0 flat shape ({root_cause, kind, namespace, ...} with
    no root_causes array). A fully-correct answer written in that shape
    directly to answer.json has no root_causes key, so it must still score
    answer_format_valid=0.0 and reward=0.0 -- exactly the schema
    `_instruction_body()` mistakenly specified before it was removed."""
    with tempfile.TemporaryDirectory() as td:
        grade = _load_grade(Path(td))
        grade.ANSWER_PATH.write_text(json.dumps({
            "root_cause": "root-pod-xyz",
            "kind": "Pod",
            "namespace": "otel-demo",
            "reasoning": "root-pod-1 crashed",
            "propagation_chain": ["root-pod-1", "other-service"],
        }))
        result = _run(grade)
        assert result["answer_format_valid"] == 0.0, result
        assert result["reward"] == 0.0, result


# ---------------------------------------------------------------------------
# Transcript recovery (reward=0.0 on new schema, regardless of correctness)
# ---------------------------------------------------------------------------

def test_recovery_correct_answer_scores_zero():
    """Recovery fires with correct answer -> reward=0.0 (no partial credit)."""
    with tempfile.TemporaryDirectory() as td:
        grade = _load_grade(Path(td))
        _write_transcript_new_schema(grade, [{"name": "root-pod-xyz", "kind": "Pod"}])
        result = _run(grade)
        assert result["answer_format_valid"] == 0.0, result
        assert result["answer_recovered_from_transcript"] == 1.0, result
        assert result["name_match"] == 1.0, result
        assert result["kind_match"] == 1.0, result
        assert result["reward"] == 0.0, result  # no partial credit


def test_recovery_wrong_answer_scores_zero():
    """Recovery fires with wrong answer -> reward stays 0.0."""
    with tempfile.TemporaryDirectory() as td:
        grade = _load_grade(Path(td))
        _write_transcript_new_schema(grade, [{"name": "totally-wrong", "kind": "Pod"}])
        result = _run(grade)
        assert result["answer_recovered_from_transcript"] == 1.0, result
        assert result["reward"] == 0.0, result


def test_legacy_recovery_correct_answer_scores_zero():
    """Legacy flat schema in transcript recovers and converts to new schema;
    reward still 0.0."""
    with tempfile.TemporaryDirectory() as td:
        grade = _load_grade(Path(td))
        _write_transcript_legacy(grade, "root-pod-xyz", "Pod")
        result = _run(grade)
        assert result["answer_recovered_from_transcript"] == 1.0, result
        assert result["name_match"] == 1.0, result
        assert result["reward"] == 0.0, result


def test_recovery_prints_fallback_notice():
    """Stdout must say so when recovery fires and show the recovered blob."""
    with tempfile.TemporaryDirectory() as td:
        grade = _load_grade(Path(td))
        _write_transcript_new_schema(grade, [{"name": "root-pod-xyz", "kind": "Pod"}])
        result, output = _run_capturing_stdout(grade)
        assert result["answer_recovered_from_transcript"] == 1.0, result
        assert "NOT FOUND on disk" in output, output
        assert "recovered from the agent's transcript" in output, output


# ---------------------------------------------------------------------------
# No answer at all
# ---------------------------------------------------------------------------

def test_no_answer_at_all():
    """No answer.json, no transcript -> reward=0.0; applicability flags still
    reflect ground truth, not the absence of an answer."""
    with tempfile.TemporaryDirectory() as td:
        grade = _load_grade(Path(td))
        result = _run(grade)
        assert result["reward"] == 0.0, result
        assert result["answer_format_valid"] == 0.0, result
        assert result["answer_recovered_from_transcript"] == 0.0, result
        # MINIMAL_GT has no namespace on root cause, but does have propagations
        assert result["namespace_applicable"] == 0.0, result
        assert result["chain_applicable"] == 1.0, result


def test_no_answer_with_namespace_gt():
    """namespace_applicable must read from GT even when no answer is present."""
    with tempfile.TemporaryDirectory() as td:
        grade = _load_grade(Path(td), ground_truth_yaml=NAMESPACE_GT)
        result = _run(grade)
        assert result["namespace_applicable"] == 1.0, result
        assert result["reward"] == 0.0, result


def test_no_answer_prints_not_found_notice():
    with tempfile.TemporaryDirectory() as td:
        grade = _load_grade(Path(td))
        _, output = _run_capturing_stdout(grade)
        assert "NOT FOUND on disk" in output, output
        assert "no recoverable answer" in output, output


# ---------------------------------------------------------------------------
# Namespace matching
# ---------------------------------------------------------------------------

def test_namespace_match_correct():
    with tempfile.TemporaryDirectory() as td:
        grade = _load_grade(Path(td), ground_truth_yaml=NAMESPACE_GT)
        _write_answer(grade, [{"name": "root-pod-xyz", "kind": "Pod",
                                "namespace": "otel-demo"}],
                      reasoning="root-pod-1 crashed")
        result = _run(grade)
        assert result["namespace_match"] == 1.0, result
        assert result["namespace_applicable"] == 1.0, result
        assert result["reward"] == 1.0, result


def test_namespace_wrong_blocks_reward():
    """Correct name+kind but wrong namespace -> name_match=kind_match=1 but
    namespace_match=0 -> reward=0."""
    with tempfile.TemporaryDirectory() as td:
        grade = _load_grade(Path(td), ground_truth_yaml=NAMESPACE_GT)
        _write_answer(grade, [{"name": "root-pod-xyz", "kind": "Pod",
                                "namespace": "chaos-mesh"}],
                      reasoning="root-pod-1 crashed")
        result = _run(grade)
        assert result["name_match"] == 1.0, result
        assert result["kind_match"] == 1.0, result
        assert result["namespace_match"] == 0.0, result
        assert result["reward"] == 0.0, result


def test_namespace_zero_when_gt_has_no_namespace():
    """Root cause group in MINIMAL_GT has no namespace -> namespace_applicable=0.0
    -> namespace_match=0.0 (nothing was checked; no vacuous pass).
    Conditioned rate = mean(namespace_match) / mean(namespace_applicable)."""
    with tempfile.TemporaryDirectory() as td:
        grade = _load_grade(Path(td))
        _write_answer(grade, [{"name": "root-pod-xyz", "kind": "Pod"}],
                      reasoning="root-pod-1 crashed")
        result = _run(grade)
        assert result["namespace_applicable"] == 0.0, result
        assert result["namespace_match"] == 0.0, result
        assert result["reward"] == 1.0, result


# ---------------------------------------------------------------------------
# Chain metrics
# ---------------------------------------------------------------------------

def test_chain_applicable_zero_when_no_propagations():
    """NO_PROPAGATIONS_GT has no propagations -> chain_applicable=0.0.
    chain_head_correct, edge_coverage, resolution_rate must be 0.0 (not None,
    Harbor rejects None)."""
    with tempfile.TemporaryDirectory() as td:
        grade = _load_grade(Path(td), ground_truth_yaml=NO_PROPAGATIONS_GT)
        _write_answer(grade, [{"name": "root-pod-xyz", "kind": "Pod"}],
                      reasoning="root-pod-1 crashed",
                      propagation_chain=["root-pod-1"])
        result = _run(grade)
        assert result["reward"] == 1.0, result
        assert result["chain_applicable"] == 0.0, result
        # Harbor requires all values to be float; 0.0 signals "not applicable"
        assert isinstance(result["chain_head_correct"], float), result
        assert isinstance(result["propagation_edge_coverage"], float), result
        assert isinstance(result["chain_resolution_rate"], float), result


def test_chain_head_correct_when_propagations_exist():
    """MINIMAL_GT has propagations; oracle chain starts at root -> head=1.0."""
    with tempfile.TemporaryDirectory() as td:
        grade = _load_grade(Path(td))
        _write_answer(grade, [{"name": "root-pod-xyz", "kind": "Pod"}],
                      reasoning="root-pod-1 crashed",
                      propagation_chain=["root-pod-1", "other-service"])
        result = _run(grade)
        assert result["chain_applicable"] == 1.0, result
        assert result["chain_head_correct"] == 1.0, result
        assert result["propagation_edge_coverage"] == 1.0, result
        assert result["chain_resolution_rate"] == 1.0, result


def test_missing_chain_is_applicable_zero_not_vacuous():
    """Agent omits propagation_chain but GT has propagations -> applicable
    failure (chain_applicable=1.0, edge_coverage=0.0), not a vacuous pass."""
    with tempfile.TemporaryDirectory() as td:
        grade = _load_grade(Path(td))
        _write_answer(grade, [{"name": "root-pod-xyz", "kind": "Pod"}],
                      reasoning="root-pod-1 crashed")
        result = _run(grade)
        assert result["chain_applicable"] == 1.0, result
        assert result["propagation_edge_coverage"] == 0.0, result


# ---------------------------------------------------------------------------
# Reasoning presence (token resolution, not char count)
# ---------------------------------------------------------------------------

def test_reasoning_present_when_token_resolves():
    """A reasoning string containing a token that resolves to a GT group
    -> reasoning_present=1.0."""
    with tempfile.TemporaryDirectory() as td:
        grade = _load_grade(Path(td))
        # "root-pod-1" resolves exactly to the group id
        _write_answer(grade, [{"name": "root-pod-xyz", "kind": "Pod"}],
                      reasoning="root-pod-1 is the root cause of the incident")
        result = _run(grade)
        assert result["reasoning_present"] == 1.0, result


def test_reasoning_absent_when_no_token_resolves():
    """A reasoning string with no tokens that resolve to any GT group
    -> reasoning_present=0.0."""
    with tempfile.TemporaryDirectory() as td:
        grade = _load_grade(Path(td))
        _write_answer(grade, [{"name": "root-pod-xyz", "kind": "Pod"}],
                      reasoning="something completely unrelated happened")
        result = _run(grade)
        assert result["reasoning_present"] == 0.0, result


def test_reasoning_absent_when_missing():
    """Missing reasoning field -> reasoning_present=0.0."""
    with tempfile.TemporaryDirectory() as td:
        grade = _load_grade(Path(td))
        _write_answer(grade, [{"name": "root-pod-xyz", "kind": "Pod"}])
        result = _run(grade)
        assert result["reasoning_present"] == 0.0, result


# ---------------------------------------------------------------------------
# All reward dict values must be float (Harbor requirement)
# ---------------------------------------------------------------------------

def test_all_reward_values_are_float():
    """Harbor's verifier rejects any non-numeric reward value, including None.
    Verify every key in the reward dict is a finite float or int."""
    import math
    with tempfile.TemporaryDirectory() as td:
        grade = _load_grade(Path(td))
        _write_answer(grade, [{"name": "root-pod-xyz", "kind": "Pod"}],
                      reasoning="root-pod-1 crashed",
                      propagation_chain=["root-pod-1", "other-service"])
        result = _run(grade)
        for k, v in result.items():
            assert isinstance(v, (int, float)), f"{k}={v!r} is not numeric"
            assert math.isfinite(v), f"{k}={v!r} is not finite"


def test_all_reward_values_are_float_no_propagations():
    """Same check for NO_PROPAGATIONS_GT where chain metrics are inapplicable
    (must still be 0.0, not None)."""
    import math
    with tempfile.TemporaryDirectory() as td:
        grade = _load_grade(Path(td), ground_truth_yaml=NO_PROPAGATIONS_GT)
        _write_answer(grade, [{"name": "root-pod-xyz", "kind": "Pod"}],
                      reasoning="root-pod-1 crashed")
        result = _run(grade)
        for k, v in result.items():
            assert isinstance(v, (int, float)), f"{k}={v!r} is not numeric"
            assert math.isfinite(v), f"{k}={v!r} is not finite"


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

_TESTS = [
    test_single_correct_entity_full_reward,
    test_correct_plus_one_spurious_halves_reward,
    test_correct_plus_two_spurious_thirds_reward,
    test_wrong_entity_only_scores_zero,
    test_duplicate_submission_second_in_denominator,
    test_multi_rc_one_of_two_correct_fires_recall_gate,
    test_multi_rc_both_correct_full_reward,
    test_alias_sibling_service_matches_pod_root_cause,
    test_workload_equiv_deployment_matches_pod_root,
    test_workload_equiv_bare_pod_name_matches_root,
    test_authored_alias_service_is_respected,
    test_workload_equiv_does_not_admit_victim_via_new_path,
    test_workload_equiv_foreign_workload_does_not_match,
    test_chain_resolver_root_identity_resolves_to_root_head,
    test_deployment_bare_name_matches_widened_filter,
    test_deployment_pod_name_also_matches_widened_filter,
    test_unrelated_name_does_not_match_widened_filter,
    test_missing_root_causes_format_invalid,
    test_wrong_schema_version_format_invalid,
    test_missing_schema_version_still_valid,
    test_legacy_flat_schema_on_disk_format_invalid,
    test_recovery_correct_answer_scores_zero,
    test_recovery_wrong_answer_scores_zero,
    test_legacy_recovery_correct_answer_scores_zero,
    test_recovery_prints_fallback_notice,
    test_no_answer_at_all,
    test_no_answer_with_namespace_gt,
    test_no_answer_prints_not_found_notice,
    test_namespace_match_correct,
    test_namespace_wrong_blocks_reward,
    test_namespace_zero_when_gt_has_no_namespace,
    test_chain_applicable_zero_when_no_propagations,
    test_chain_head_correct_when_propagations_exist,
    test_missing_chain_is_applicable_zero_not_vacuous,
    test_reasoning_present_when_token_resolves,
    test_reasoning_absent_when_no_token_resolves,
    test_reasoning_absent_when_missing,
    test_all_reward_values_are_float,
    test_all_reward_values_are_float_no_propagations,
]


def main() -> int:
    failures = 0
    for test in _TESTS:
        try:
            test()
        except AssertionError as e:
            failures += 1
            print(f"FAIL: {test.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            failures += 1
            print(f"ERROR: {test.__name__}: {e!r}")
        else:
            print(f"PASS: {test.__name__}")
    print(f"\n{len(_TESTS) - failures}/{len(_TESTS)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())