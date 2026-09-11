#!/usr/bin/env python3
"""
Regression tests for the itbench-lite grader
(src/itbench_lite/task-template/tests/grade.py).

Focused on the transcript-recovery fallback and its reward penalty: as of
writing, every job on disk (oracle and real agent runs alike) has
answer_recovered_from_transcript == 0.0 -- the fallback has never actually
fired in practice, so it had zero regression coverage. These tests exercise
it directly against a fresh, monkeypatched copy of the grader module (no
real /workspace or /logs paths are touched).

Runnable standalone (no pytest dependency, matching this adapter's minimal
runtime):
    python3 adapters/itbench-lite/tests/test_grade.py
Also pytest-discoverable (test_-prefixed functions) if pytest is ever added
to this repo's dev dependencies.
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
    / "itbench_lite"
    / "task-template"
    / "tests"
    / "grade.py"
)

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

NO_PROPAGATIONS_GT = r"""
groups:
  - id: root-pod-1
    kind: Pod
    filter:
      - root-pod-.*
    root_cause: true
"""

DANGLING_ID_GT = r"""
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
  - source: other-service-1
    target: undefined-downstream-1
    condition: other service misbehaves
    effect: undefined downstream errors
"""

# A Deployment-kind root cause reusing the Pod-oriented "<literal>-.*"
# filter convention -- real Deployments never get a random hash suffix
# (only their owned Pods/ReplicaSets do), so the bare literal must also
# satisfy this filter for a factually correct, telemetry-grounded answer to
# be gradeable at all (see matching.py's name_matches suffix relaxation).
SUFFIX_GT = r"""
groups:
  - id: my-app-deployment
    kind: Deployment
    filter:
      - my-app-.*
    root_cause: true
"""

NAMESPACE_GT = r"""
groups:
  - id: root-pod-1
    kind: Pod
    namespace: otel-demo
    filter:
      - root-pod-.*
    root_cause: true
"""

# A Chaos Mesh root cause reusing the OTel Demo app's full "service.name"
# form (e.g. "adservice") in its filter, while the real k8s object short
# name drops the "service" suffix (e.g. "ad") -- scenario-29-class bug this
# session's audit found: no telemetry-grounded answer naming the real CR
# (e.g. "otel-demo-ad-jvm-return-2xnmj") could otherwise pass.
SERVICE_SUFFIX_GT = r"""
groups:
  - id: jvm-return-otel-demo-ad-1
    kind: JVMChaos
    filter:
      - .*adservice
    root_cause: true
"""


def _load_grade(tmp_path: Path, ground_truth_yaml: str = MINIMAL_GT):
    """Import a fresh grade module instance with its path constants
    monkeypatched into tmp_path, so each test is isolated from the real
    filesystem and from other tests."""
    # grade.py does `from matching import ...` (sibling import, resolved via
    # Harbor's real cwd=/tests/ at grading time) -- put its directory on
    # sys.path so that import resolves under this file-location loader too.
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
    """Like _run, but also returns everything grade.main() printed, so
    tests can assert on the human-readable answer echo in
    test-stdout.txt (not just the machine-readable reward.json)."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        grade.main()
    return json.loads(grade.REWARD_PATH.read_text()), buf.getvalue()


def _write_transcript(grade, root_cause: str, kind: str) -> None:
    """Simulate an agent that stated its conclusion in chat but never
    wrote /workspace/answer.json."""
    transcript = grade.AGENT_LOGS_DIR / "session.jsonl"
    entry = {
        "message": {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": f'My conclusion: {{"root_cause": "{root_cause}", "kind": "{kind}"}}',
                }
            ],
        }
    }
    transcript.write_text(json.dumps(entry) + "\n")


def test_normal_answer_full_credit():
    """Sanity check: the ordinary file-write path is unaffected by the
    recovery penalty."""
    with tempfile.TemporaryDirectory() as td:
        grade = _load_grade(Path(td))
        grade.ANSWER_PATH.write_text(
            json.dumps(
                {
                    "root_cause": "root-pod-1",
                    "kind": "Pod",
                    "reasoning": "x" * 25,
                    "propagation_chain": ["root-pod-1", "other-service-1"],
                }
            )
        )
        result = _run(grade)
        assert result["answer_format_valid"] == 1.0, result
        assert result["answer_recovered_from_transcript"] == 0.0, result
        assert result["reward"] == 1.0, result
        # Ground truth has real data for all three -- these must read as
        # genuine checks, not vacuous passes.
        assert result["namespace_applicable"] == 0.0, result  # MINIMAL_GT's root has no namespace field
        assert result["chain_proximity_applicable"] == 1.0, result
        assert result["propagation_chain_coverage_applicable"] == 1.0, result


def test_recovery_correct_answer_halved():
    """Agent never writes answer.json but states the right answer in
    chat -- recovery should fire, and reward should be halved (0.5), not
    full credit and not zero."""
    with tempfile.TemporaryDirectory() as td:
        grade = _load_grade(Path(td))
        _write_transcript(grade, "root-pod-1", "Pod")
        result = _run(grade)
        assert result["answer_format_valid"] == 0.0, result
        assert result["answer_recovered_from_transcript"] == 1.0, result
        assert result["kind_match"] == 1.0, result
        assert result["name_match"] == 1.0, result
        assert result["reward"] == 0.5, result


def test_recovery_wrong_answer_scores_zero():
    """Recovery firing doesn't rescue a wrong answer -- 0.0 * 0.5 is still
    0.0, nothing to halve."""
    with tempfile.TemporaryDirectory() as td:
        grade = _load_grade(Path(td))
        _write_transcript(grade, "other-service-1", "Service")
        result = _run(grade)
        assert result["answer_recovered_from_transcript"] == 1.0, result
        assert result["kind_match"] == 0.0, result
        assert result["name_match"] == 0.0, result
        assert result["reward"] == 0.0, result


def test_no_answer_at_all():
    """No answer.json, no transcript to recover from -- clean 0.0, no
    exceptions. Applicability must still reflect MINIMAL_GT's real ground
    truth (no namespace field, but real propagations) -- a total agent
    failure to answer must not be reported identically to a scenario with
    nothing to check (see test_missing_propagation_chain_is_applicable_zero_not_vacuous
    for the same principle when an answer IS present)."""
    with tempfile.TemporaryDirectory() as td:
        grade = _load_grade(Path(td))
        result = _run(grade)
        assert result["answer_format_valid"] == 0.0, result
        assert result["answer_recovered_from_transcript"] == 0.0, result
        assert result["reward"] == 0.0, result
        assert result["namespace_applicable"] == 0.0, result  # MINIMAL_GT's root has no namespace field
        assert result["chain_proximity_applicable"] == 1.0, result
        assert result["propagation_chain_coverage_applicable"] == 1.0, result


def test_no_answer_at_all_with_namespace_in_ground_truth():
    """Same as test_no_answer_at_all but against NAMESPACE_GT -- proves
    namespace_applicable is computed from the ground truth even when there's
    no answer to grade, not just hardcoded to 0.0."""
    with tempfile.TemporaryDirectory() as td:
        grade = _load_grade(Path(td), ground_truth_yaml=NAMESPACE_GT)
        result = _run(grade)
        assert result["reward"] == 0.0, result
        assert result["namespace_applicable"] == 1.0, result


def test_recovery_prints_fallback_notice():
    """When answer.json is missing but recovery fires, stdout must say so
    explicitly (not silently print the recovered blob as if it were the
    real file) and still include the recovered content."""
    with tempfile.TemporaryDirectory() as td:
        grade = _load_grade(Path(td))
        _write_transcript(grade, "root-pod-1", "Pod")
        result, output = _run_capturing_stdout(grade)
        assert result["answer_recovered_from_transcript"] == 1.0, result
        assert "NOT FOUND on disk" in output, output
        assert "recovered from the agent's transcript" in output, output
        assert '"root_cause": "root-pod-1"' in output, output
        assert '"kind": "Pod"' in output, output


def test_no_answer_at_all_prints_not_found_notice():
    """No answer.json and nothing recoverable from the transcript either --
    stdout must say so explicitly instead of staying silent about the
    missing answer."""
    with tempfile.TemporaryDirectory() as td:
        grade = _load_grade(Path(td))
        _, output = _run_capturing_stdout(grade)
        assert "NOT FOUND on disk" in output, output
        assert "no recoverable answer" in output, output


def test_no_propagations_is_vacuous_pass_not_failure():
    """Ground truth with no propagations at all (a single non-cascading
    fault) has nothing for chain_proximity/propagation_chain_coverage to
    check the answer against -- that should score as a vacuous 1.0, not a
    0.0 failure the answer had no way to avoid. reward is untouched either
    way (these metrics never gate it)."""
    with tempfile.TemporaryDirectory() as td:
        grade = _load_grade(Path(td), ground_truth_yaml=NO_PROPAGATIONS_GT)
        grade.ANSWER_PATH.write_text(
            json.dumps(
                {
                    "root_cause": "root-pod-1",
                    "kind": "Pod",
                    "reasoning": "x" * 25,
                    "propagation_chain": ["root-pod-1"],
                }
            )
        )
        result = _run(grade)
        assert result["reward"] == 1.0, result
        assert result["chain_proximity"] == 1.0, result
        assert result["propagation_chain_coverage"] == 1.0, result
        assert result["chain_proximity_applicable"] == 0.0, result
        assert result["propagation_chain_coverage_applicable"] == 0.0, result


def test_dangling_propagation_id_excluded_but_defined_ids_still_checked():
    """A propagation edge referencing an id with no defined group (e.g. a
    typo'd id in the scenario data) shouldn't be held against the answer --
    but an otherwise-incomplete answer that also fails to cover a
    well-defined id must still score less than perfect coverage. Proves the
    fix excludes only the undefined id, not the whole scenario."""
    with tempfile.TemporaryDirectory() as td:
        grade = _load_grade(Path(td), ground_truth_yaml=DANGLING_ID_GT)
        grade.ANSWER_PATH.write_text(
            json.dumps(
                {
                    "root_cause": "root-pod-1",
                    "kind": "Pod",
                    "reasoning": "x" * 25,
                    # Only covers the root -- doesn't mention other-service-1
                    # (well-defined) or undefined-downstream-1 (dangling).
                    "propagation_chain": ["root-pod-1"],
                }
            )
        )
        result = _run(grade)
        assert result["reward"] == 1.0, result
        # 1 of 2 verifiable ids covered (root-pod-1 yes, other-service-1 no);
        # undefined-downstream-1 excluded from both numerator and denominator.
        assert result["propagation_chain_coverage"] == 0.5, result


def test_missing_propagation_chain_is_applicable_zero_not_vacuous():
    """The agent omitting propagation_chain when the ground truth DOES have
    real propagations to check is a real, applicable failure (0.0) -- not a
    vacuous pass. Applicability is a property of the ground truth, not of
    whether the agent bothered to answer."""
    with tempfile.TemporaryDirectory() as td:
        grade = _load_grade(Path(td))  # MINIMAL_GT has real propagations
        grade.ANSWER_PATH.write_text(
            json.dumps({"root_cause": "root-pod-1", "kind": "Pod", "reasoning": "x" * 25})
        )
        result = _run(grade)
        assert result["propagation_chain_coverage"] == 0.0, result
        assert result["propagation_chain_coverage_applicable"] == 1.0, result


def test_suffix_relaxation_matches_bare_unsuffixed_name():
    """A factually correct answer naming the real (suffix-less) Deployment
    must satisfy a filter written for the suffixed-Pod convention -- the
    scenario-105 bug this session's audit found: only the internal synthetic
    id used to pass "my-app-.*", not the real bare resource name."""
    with tempfile.TemporaryDirectory() as td:
        grade = _load_grade(Path(td), ground_truth_yaml=SUFFIX_GT)
        grade.ANSWER_PATH.write_text(
            json.dumps({"root_cause": "my-app", "kind": "Deployment", "reasoning": "x" * 25})
        )
        result = _run(grade)
        assert result["name_match"] == 1.0, result
        assert result["reward"] == 1.0, result


def test_suffix_relaxation_does_not_widen_to_unrelated_names():
    """The relaxation only adds the zero-length-suffix case -- it must not
    make the filter match some other, unrelated entity."""
    with tempfile.TemporaryDirectory() as td:
        grade = _load_grade(Path(td), ground_truth_yaml=SUFFIX_GT)
        grade.ANSWER_PATH.write_text(
            json.dumps({"root_cause": "totally-different-service", "kind": "Deployment", "reasoning": "x" * 25})
        )
        result = _run(grade)
        assert result["name_match"] == 0.0, result
        assert result["reward"] == 0.0, result


def test_service_suffix_relaxation_matches_real_chaos_object_name():
    """A factually correct answer naming the real, telemetry-grounded Chaos
    Mesh object (whose k8s short name drops the app's descriptive "service"
    suffix) must satisfy a filter written using that full service name."""
    with tempfile.TemporaryDirectory() as td:
        grade = _load_grade(Path(td), ground_truth_yaml=SERVICE_SUFFIX_GT)
        grade.ANSWER_PATH.write_text(
            json.dumps(
                {
                    "root_cause": "otel-demo-ad-jvm-return-2xnmj",
                    "kind": "JVMChaos",
                    "reasoning": "x" * 25,
                }
            )
        )
        result = _run(grade)
        assert result["name_match"] == 1.0, result
        assert result["reward"] == 1.0, result


def test_service_suffix_relaxation_does_not_widen_to_unrelated_names():
    """The relaxation must require "ad" as a whole hyphen-delimited token,
    not a raw substring -- must not match a name that merely contains the
    same letters (e.g. "ad" inside "load")."""
    with tempfile.TemporaryDirectory() as td:
        grade = _load_grade(Path(td), ground_truth_yaml=SERVICE_SUFFIX_GT)
        grade.ANSWER_PATH.write_text(
            json.dumps({"root_cause": "load-generator-pod-1", "kind": "JVMChaos", "reasoning": "x" * 25})
        )
        result = _run(grade)
        assert result["name_match"] == 0.0, result
        assert result["reward"] == 0.0, result


def test_namespace_match_correct():
    with tempfile.TemporaryDirectory() as td:
        grade = _load_grade(Path(td), ground_truth_yaml=NAMESPACE_GT)
        grade.ANSWER_PATH.write_text(
            json.dumps(
                {
                    "root_cause": "root-pod-1",
                    "kind": "Pod",
                    "namespace": "otel-demo",
                    "reasoning": "x" * 25,
                }
            )
        )
        result = _run(grade)
        assert result["namespace_match"] == 1.0, result
        assert result["namespace_applicable"] == 1.0, result
        assert result["reward"] == 1.0, result


def test_namespace_match_wrong_namespace_fails_reward():
    """An otherwise-correct answer in the wrong namespace must not score
    reward=1.0 -- namespace_match gates reward alongside kind_match/
    name_match."""
    with tempfile.TemporaryDirectory() as td:
        grade = _load_grade(Path(td), ground_truth_yaml=NAMESPACE_GT)
        grade.ANSWER_PATH.write_text(
            json.dumps(
                {
                    "root_cause": "root-pod-1",
                    "kind": "Pod",
                    "namespace": "chaos-mesh",
                    "reasoning": "x" * 25,
                }
            )
        )
        result = _run(grade)
        assert result["kind_match"] == 1.0, result
        assert result["name_match"] == 1.0, result
        assert result["namespace_match"] == 0.0, result
        assert result["reward"] == 0.0, result


def test_namespace_vacuous_pass_when_ground_truth_has_no_namespace():
    """MINIMAL_GT's root group has no namespace field at all (e.g. the
    root cause is itself a Namespace-kind object) -- there is nothing to
    check the answer's namespace against, so this must be a vacuous pass
    even when the answer omits namespace entirely."""
    with tempfile.TemporaryDirectory() as td:
        grade = _load_grade(Path(td))
        grade.ANSWER_PATH.write_text(
            json.dumps({"root_cause": "root-pod-1", "kind": "Pod", "reasoning": "x" * 25})
        )
        result = _run(grade)
        assert result["namespace_match"] == 1.0, result
        assert result["namespace_applicable"] == 0.0, result
        assert result["reward"] == 1.0, result


_TESTS = [
    test_normal_answer_full_credit,
    test_recovery_correct_answer_halved,
    test_recovery_wrong_answer_scores_zero,
    test_no_answer_at_all,
    test_no_answer_at_all_with_namespace_in_ground_truth,
    test_recovery_prints_fallback_notice,
    test_no_answer_at_all_prints_not_found_notice,
    test_no_propagations_is_vacuous_pass_not_failure,
    test_dangling_propagation_id_excluded_but_defined_ids_still_checked,
    test_missing_propagation_chain_is_applicable_zero_not_vacuous,
    test_suffix_relaxation_matches_bare_unsuffixed_name,
    test_suffix_relaxation_does_not_widen_to_unrelated_names,
    test_service_suffix_relaxation_matches_real_chaos_object_name,
    test_service_suffix_relaxation_does_not_widen_to_unrelated_names,
    test_namespace_match_correct,
    test_namespace_match_wrong_namespace_fails_reward,
    test_namespace_vacuous_pass_when_ground_truth_has_no_namespace,
]


def main() -> int:
    failures = 0
    for test in _TESTS:
        try:
            test()
        except AssertionError as e:
            failures += 1
            print(f"FAIL: {test.__name__}: {e}")
        except Exception as e:  # noqa: BLE001 -- report any error as a test failure
            failures += 1
            print(f"ERROR: {test.__name__}: {e!r}")
        else:
            print(f"PASS: {test.__name__}")
    print(f"\n{len(_TESTS) - failures}/{len(_TESTS)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
