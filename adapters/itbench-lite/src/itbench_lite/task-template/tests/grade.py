#!/usr/bin/env python3
"""
Grader for itbench-lite tasks.

Reads the agent's /workspace/answer.json, compares it against
tests/ground_truth.yaml (copied in at task-generation time, grader-only --
never present under environment/), and writes a multi-metric reward to
/logs/verifier/reward.json.

Scoring contract (see adapter's approved plan):
- answer_format_valid: 1.0 iff /workspace/answer.json exists, parses as JSON,
  and has both "root_cause" and "kind" string keys. This is a mechanical
  signal ("did the agent actually write the file") and is NOT set to 1.0 by
  the transcript-recovery fallback below.
- answer_recovered_from_transcript: 1.0 iff /workspace/answer.json was
  missing/malformed but a root_cause/kind JSON blob was recovered from the
  agent's own final transcript message under /logs/agent/ instead. This is a
  best-effort safety net purely for visibility (kind_match/name_match/
  reasoning_present/propagation_chain_coverage still reflect the agent's
  actual stated answer instead of a hard 0 for a harness-completion slip) --
  it is NOT a free pass on reward: see reward's recovery penalty below.
  Always 0.0 when answer_format_valid is 1.0. reasoning/propagation_chain
  are carried over into the recovered answer when present in the same
  recovered blob, so reasoning_present and propagation_chain_coverage still
  reflect them.
- Strict alias policy: matching is done ONLY against the root-cause group's
  own `filter` patterns. The `aliases` block is deliberately never
  consulted -- in this dataset it mixes the root cause with downstream
  propagated/symptom entities, which must not score as correct.
- Filter patterns are normally Python regexes, matched with re.search (a
  pattern shaped "<literal>-.*"/"<literal>-.+" also matches the bare
  "<literal>" with no trailing suffix -- Deployments/Services/ConfigMaps
  never get the random hash suffix that their owned Pods/ReplicaSets do, so
  a filter written for the suffixed case must not reject the unsuffixed
  real name). A pattern that isn't valid regex at all is treated as this
  dataset's glob-style "<namespace-glob>.<name-glob>" selector convention
  (e.g. "*.*" = any object of this kind, in this namespace) and matched
  with fnmatch against its name-glob segment only. See matching.py for the
  shared implementation.
- kind_match: 1.0 iff answer["kind"] equals root_group["kind"] ignoring case
  (agents commonly write "pod" for "Pod"; upstream scores this via an LLM
  judge, so case-insensitivity is the closer analogue).
- name_match: 1.0 iff answer["root_cause"] matches (re.search, with the
  suffix relaxation above) at least one of the root group's patterns.
  Patterns are its filter[] regexes; groups without a filter (e.g. a
  ConfigMap root cause) fall back to an anchored exact match on their name
  field (^<name>$).
- namespace_match: 1.0 iff answer["namespace"] equals root_group["namespace"]
  ignoring case. Vacuous 1.0 (not a failure) when the ground truth's
  root-cause group has no namespace field at all -- e.g. when the root
  cause is itself a Namespace-kind object, which has no parent namespace to
  report. namespace_applicable is 0.0 exactly on that vacuous branch (1.0
  otherwise) -- lets aggregate stats tell a real match apart from a
  scenario with nothing to check, instead of both reading as an identical
  1.0. All three *_applicable fields (namespace/chain_proximity/
  propagation_chain_coverage) are properties of the ground truth alone --
  computed the same way even when the agent supplied no answer.json and
  nothing was recoverable from its transcript, so a total non-answer on a
  scenario with real checkable content isn't misreported as vacuous.
- reward (headline): 1.0 iff kind_match == 1.0 AND name_match == 1.0 AND
  namespace_match == 1.0 AND the answer came from /workspace/answer.json
  (answer_recovered_from_transcript == 0.0). If the same otherwise-correct
  answer was only recovered from the transcript (answer_recovered_from_transcript == 1.0), reward is halved to 0.5 -- a partial-credit penalty:
  the agent reasoned correctly but did not complete the task's file-write
  requirement. A recovered answer that also fails kind_match/name_match/
  namespace_match still scores 0.0 (nothing to halve).
- chain_proximity: informational, best-effort, never affects reward. 1.0 if
  answer["root_cause"] matches the filter patterns of any group referenced by
  propagations[].source / .target (each propagation id is resolved to its
  groups[] entry and matched with the same filter-regex logic as name_match,
  since agents answer with real telemetry names, not synthetic group ids).
  Vacuous 1.0 (not a failure) when the ground truth itself has no
  propagations, or references only ids with no defined group -- there is
  nothing there for the answer to get wrong. chain_proximity_applicable is
  0.0 exactly on that vacuous branch (1.0 otherwise), same rationale as
  namespace_applicable above.
- reasoning_present: informational. 1.0 iff answer["reasoning"] is a string
  of at least 20 characters after stripping (i.e. the agent actually
  articulated what/why/how, not just where). Upstream scores reasoning via
  an LLM judge (root_cause_reasoning); this is the deterministic presence
  proxy. The agent's full raw answer (every key) is pretty-printed to
  verifier stdout (test-stdout.txt) for qualitative review -- answer.json
  itself never leaves the container, so this is the only place a human can
  see exactly what the agent produced. When recovered from the transcript
  fallback instead of a real answer.json, that's called out explicitly in
  the same stdout line rather than printed as if it were the real file; if
  neither the file nor a recoverable transcript blob exists, stdout says so
  too instead of staying silent.
- propagation_chain_coverage: informational, in [0, 1]. Fraction of the
  distinct, verifiable groups referenced by propagations[].source/.target
  (ones with a defined group -- see chain_proximity) whose filter patterns
  match at least one entry of answer["propagation_chain"] (the agent's
  declared root->symptom path). 0.0 if answer["propagation_chain"] itself is
  missing/empty (the agent's fault). Vacuous 1.0 if the ground truth has no
  propagations, or none of the referenced ids have a defined group --
  ground-truth availability is checked before the agent's own answer, so
  propagation_chain_coverage_applicable (0.0 on that vacuous branch, 1.0
  otherwise) never conflates "nothing to check" with "the agent didn't
  answer" -- those are different, orthogonal reasons the raw score can look
  similar. Deterministic analogue of upstream's propagation_chain metric;
  never affects reward.
"""

import json
import re
import sys
from pathlib import Path

from matching import find_root_cause_group, group_patterns, name_matches, normalize_ground_truth

ANSWER_PATH = Path("/workspace/answer.json")
GROUND_TRUTH_PATH = Path(__file__).resolve().parent / "ground_truth.yaml"
REWARD_DIR = Path("/logs/verifier")
REWARD_PATH = REWARD_DIR / "reward.json"
AGENT_LOGS_DIR = Path("/logs/agent")

# Matches a flat {"root_cause": ..., "kind": ...} object in either key order,
# tolerating extra whitespace/newlines. Deliberately excludes nested braces --
# the expected answer shape has no nested objects, so this stays simple and
# avoids over-matching into surrounding prose.
_ANSWER_BLOB_RE = re.compile(
    r'\{[^{}]*"root_cause"[^{}]*"kind"[^{}]*\}|\{[^{}]*"kind"[^{}]*"root_cause"[^{}]*\}'
)


def _iter_transcript_texts(agent_dir: Path):
    """Yield (mtime, text) for every assistant message found in any
    transcript file under /logs/agent/, in no particular order (caller sorts
    by mtime). Supports both JSONL session logs (role: assistant, content
    text blocks) and plain-text transcripts. Best-effort: skips any file or
    line it can't parse rather than raising.
    """
    if not agent_dir.is_dir():
        return
    for path in sorted(agent_dir.rglob("*")):
        if not path.is_file() or path.suffix not in (".jsonl", ".txt"):
            continue
        try:
            mtime = path.stat().st_mtime
            text = path.read_text(errors="ignore")
        except OSError:
            continue
        if path.suffix != ".jsonl":
            yield mtime, text
            continue
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            message = entry.get("message") if isinstance(entry, dict) else None
            if not isinstance(message, dict) or message.get("role") != "assistant":
                continue
            content = message.get("content")
            if isinstance(content, str):
                yield mtime, content
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str):
                        yield mtime, block["text"]


def recover_answer_from_transcript() -> dict | None:
    """Last-resort recovery: scan the agent's own transcript (most recent
    text first) for a root_cause/kind JSON blob it stated but never wrote to
    /workspace/answer.json. Never raises -- returns None on any failure or
    if nothing matches.
    """
    try:
        candidates = sorted(_iter_transcript_texts(AGENT_LOGS_DIR), key=lambda pair: pair[0])
    except Exception:
        return None
    for _, text in reversed(candidates):
        for blob in reversed(_ANSWER_BLOB_RE.findall(text)):
            try:
                candidate = json.loads(blob)
            except json.JSONDecodeError:
                continue
            if (
                isinstance(candidate, dict)
                and isinstance(candidate.get("root_cause"), str)
                and isinstance(candidate.get("kind"), str)
            ):
                recovered = {"root_cause": candidate["root_cause"], "kind": candidate["kind"]}
                if isinstance(candidate.get("namespace"), str):
                    recovered["namespace"] = candidate["namespace"]
                if isinstance(candidate.get("reasoning"), str):
                    recovered["reasoning"] = candidate["reasoning"]
                if isinstance(candidate.get("propagation_chain"), list):
                    recovered["propagation_chain"] = candidate["propagation_chain"]
                return recovered
    return None


def load_answer() -> tuple[dict | None, bool, bool]:
    """Returns (answer_dict_or_None, file_format_valid, recovered_from_transcript)."""
    if ANSWER_PATH.is_file():
        try:
            with ANSWER_PATH.open("r") as f:
                answer = json.load(f)
        except (json.JSONDecodeError, OSError):
            answer = None
        if isinstance(answer, dict):
            root_cause = answer.get("root_cause")
            kind = answer.get("kind")
            if isinstance(root_cause, str) and isinstance(kind, str):
                return answer, True, False

    recovered = recover_answer_from_transcript()
    if recovered is not None:
        return recovered, False, True
    return None, False, False


def compute_chain_proximity(answer_root_cause: str, ground_truth: dict) -> tuple[float, float]:
    """Best-effort, informational only. Never raises; never affects reward.

    Returns (chain_proximity, chain_proximity_applicable).

    Resolves each propagations[].source/.target id to its groups[] entry and
    matches the answer against that group's filter regexes -- agents answer
    with real telemetry names (e.g. a full pod name), which never equal the
    synthetic group ids, so exact-string comparison would always score 0.

    Vacuous pass (1.0) when the ground truth itself has no propagations to
    check the answer against (e.g. a single non-cascading fault) -- there is
    nothing here the answer could get wrong. This never grades the agent's
    own answer as correct-by-default: it only fires when the ground truth
    side has no data, not when the agent's declared root_cause is missing or
    malformed (that still can't match anything and stays at 0.0 below).

    `chain_proximity_applicable` is 0.0 exactly when that vacuous-pass
    branch fires -- so aggregate stats (e.g. "average chain_proximity") can
    distinguish a scenario with nothing to verify from a genuine match,
    instead of reading both as an identical 1.0.
    """
    try:
        propagations = ground_truth.get("propagations")
        if not isinstance(propagations, list) or not propagations:
            return 1.0, 0.0
        group_filters: dict[str, list] = {}
        for group in ground_truth.get("groups", []) or []:
            if isinstance(group, dict) and isinstance(group.get("id"), str):
                group_filters[group["id"]] = group_patterns(group)
        chain_ids: set[str] = set()
        for prop in propagations:
            if not isinstance(prop, dict):
                continue
            source = prop.get("source")
            target = prop.get("target")
            if isinstance(source, str):
                chain_ids.add(source)
            if isinstance(target, str):
                chain_ids.add(target)
        # Ids the ground truth references but never defines a group for
        # (e.g. a typo'd id) have no pattern to check against -- exclude
        # them rather than silently failing the answer on an undefined
        # reference. If nothing is verifiable at all, that's the same
        # vacuous-pass case as no propagations existing.
        verifiable_ids = [gid for gid in chain_ids if group_filters.get(gid)]
        if not verifiable_ids:
            return 1.0, 0.0
        for group_id in verifiable_ids:
            if name_matches(answer_root_cause, group_filters[group_id]):
                return 1.0, 1.0
        return 0.0, 1.0
    except Exception:
        return 0.0, 0.0


def _echo_answer(answer: dict, recovered_from_transcript: bool) -> None:
    """Surface the agent's full raw answer (every key, not just a
    hand-picked subset) to verifier stdout (test-stdout.txt) for qualitative
    review -- answer.json itself never leaves the container, so this is the
    only place a human can see what the agent actually produced.

    If /workspace/answer.json was missing/malformed, `answer` is instead the
    best-effort fallback recovered from the agent's own transcript (see
    recover_answer_from_transcript()) -- say so explicitly rather than
    silently printing it as if it were the real file, since that fallback
    caps reward at 0.5 (see main()).
    """
    if recovered_from_transcript:
        print(
            "agent answer.json: NOT FOUND on disk -- showing the best-effort "
            "fallback recovered from the agent's transcript instead "
            f"(reward is capped at 0.5 for this):\n{json.dumps(answer, indent=2)}\n"
        )
    else:
        print(f"agent answer.json:\n{json.dumps(answer, indent=2)}\n")


def compute_reasoning_present(answer: dict) -> float:
    """Informational only. 1.0 iff the answer carries a substantive
    free-text reasoning field (>= 20 chars after stripping)."""
    reasoning = answer.get("reasoning")
    return 1.0 if isinstance(reasoning, str) and len(reasoning.strip()) >= 20 else 0.0


def compute_propagation_chain_coverage(answer: dict, ground_truth: dict) -> tuple[float, float]:
    """Informational only. Never raises; never affects reward.

    Returns (propagation_chain_coverage, propagation_chain_coverage_applicable).

    Fraction of the distinct, verifiable groups referenced by
    propagations[].source/.target whose filter patterns match at least one
    entry of the agent's declared propagation_chain (matched with the same
    filter-regex logic as name_match). 0.0 when the agent's own
    propagation_chain field is missing/empty -- that's the agent's fault,
    not the ground truth's. Vacuous pass (1.0) when the ground truth itself
    has no propagations, or references only ids with no defined group (e.g.
    a typo'd id) -- there is nothing verifiable there for the answer to get
    wrong; referenced ids that ARE defined still get checked normally, so an
    otherwise-incomplete answer can't ride a typo'd id to a free pass.

    Ground-truth availability is checked before anything about the agent's
    own answer, so `propagation_chain_coverage_applicable` reflects only
    whether the ground truth had something real to check -- not whether the
    agent happened to supply a usable propagation_chain (those are
    orthogonal: a missing chain is a real, applicable 0.0, not a vacuous
    pass).
    """
    try:
        propagations = ground_truth.get("propagations")
        if not isinstance(propagations, list) or not propagations:
            return 1.0, 0.0
        group_filters: dict[str, list] = {}
        for group in ground_truth.get("groups", []) or []:
            if isinstance(group, dict) and isinstance(group.get("id"), str):
                group_filters[group["id"]] = group_patterns(group)
        chain_ids: list[str] = []
        for prop in propagations:
            if not isinstance(prop, dict):
                continue
            for key in ("source", "target"):
                value = prop.get(key)
                if isinstance(value, str) and value not in chain_ids:
                    chain_ids.append(value)
        verifiable_ids = [gid for gid in chain_ids if group_filters.get(gid)]
        if not verifiable_ids:
            return 1.0, 0.0

        # From here on there IS something real to check (applicable=1.0)
        # regardless of whether the agent's own chain field is present.
        chain = answer.get("propagation_chain")
        if not isinstance(chain, list):
            return 0.0, 1.0
        entries = [e for e in chain if isinstance(e, str) and e.strip()]
        if not entries:
            return 0.0, 1.0
        covered = sum(
            1
            for group_id in verifiable_ids
            if any(name_matches(entry, group_filters[group_id]) for entry in entries)
        )
        return covered / len(verifiable_ids), 1.0
    except Exception:
        return 0.0, 0.0


def _ground_truth_applicability(ground_truth: dict) -> tuple[float, float, float]:
    """Ground-truth-only applicability for namespace/chain-proximity/
    propagation-chain-coverage -- independent of whether the agent supplied
    any answer at all. Used by the no-answer branch of main() so that a
    scenario with real, checkable ground truth (a namespace to match, a real
    propagation chain) isn't misreported as vacuous just because the agent
    never wrote answer.json; applicability is a property of the ground
    truth, not of the agent's completion. Never raises -- falls back to
    all-0.0 (can't determine applicability) on malformed ground truth, same
    as the "no root_cause group" branch below.
    """
    try:
        root_group = find_root_cause_group(ground_truth)
        expected_namespace = root_group.get("namespace") if root_group else None
        namespace_applicable = 1.0 if isinstance(expected_namespace, str) and expected_namespace.strip() else 0.0
    except Exception:
        namespace_applicable = 0.0
    # The answer argument only affects the score half of these tuples, never
    # the applicable half -- passing an empty/unmatchable placeholder is safe.
    _, chain_proximity_applicable = compute_chain_proximity("", ground_truth)
    _, propagation_chain_coverage_applicable = compute_propagation_chain_coverage({}, ground_truth)
    return namespace_applicable, chain_proximity_applicable, propagation_chain_coverage_applicable


def main() -> int:
    import yaml  # imported lazily so a missing dependency doesn't mask other errors

    REWARD_DIR.mkdir(parents=True, exist_ok=True)

    answer, answer_format_valid, recovered_from_transcript = load_answer()

    if answer is None:
        namespace_applicable = chain_proximity_applicable = propagation_chain_coverage_applicable = 0.0
        try:
            with GROUND_TRUTH_PATH.open("r") as f:
                ground_truth = normalize_ground_truth(yaml.safe_load(f))
            namespace_applicable, chain_proximity_applicable, propagation_chain_coverage_applicable = (
                _ground_truth_applicability(ground_truth)
            )
        except Exception:
            pass  # keep the all-0.0 fallback -- still emit a well-formed reward file
        result = {
            "reward": 0.0,
            "kind_match": 0.0,
            "name_match": 0.0,
            "namespace_match": 0.0,
            "namespace_applicable": namespace_applicable,
            "answer_format_valid": 0.0,
            "answer_recovered_from_transcript": 0.0,
            "chain_proximity": 0.0,
            "chain_proximity_applicable": chain_proximity_applicable,
            "reasoning_present": 0.0,
            "propagation_chain_coverage": 0.0,
            "propagation_chain_coverage_applicable": propagation_chain_coverage_applicable,
        }
        with REWARD_PATH.open("w") as f:
            json.dump(result, f)
        print(
            "agent answer.json: NOT FOUND on disk, and no recoverable answer "
            "was found in the agent's transcript either.\n"
        )
        print(f"answer_format_valid=0: {result}")
        return 0

    with GROUND_TRUTH_PATH.open("r") as f:
        ground_truth = normalize_ground_truth(yaml.safe_load(f))

    root_group = find_root_cause_group(ground_truth)
    if root_group is None:
        # Malformed ground truth is a grader/data bug, not an agent failure --
        # still emit a well-formed reward file rather than crashing.
        # namespace_applicable genuinely requires a root group (there's no
        # namespace to check without one), but chain_proximity/
        # propagation_chain_coverage applicability only depend on
        # propagations[]/groups[], which can still be well-formed even when
        # no group is marked root_cause: true -- compute them for real
        # instead of hardcoding a vacuous 0.0.
        _, chain_proximity_applicable, propagation_chain_coverage_applicable = _ground_truth_applicability(
            ground_truth
        )
        result = {
            "reward": 0.0,
            "kind_match": 0.0,
            "name_match": 0.0,
            "namespace_match": 0.0,
            "namespace_applicable": 0.0,
            "answer_format_valid": 1.0 if answer_format_valid else 0.0,
            "answer_recovered_from_transcript": 1.0 if recovered_from_transcript else 0.0,
            "chain_proximity": 0.0,
            "chain_proximity_applicable": chain_proximity_applicable,
            "reasoning_present": compute_reasoning_present(answer),
            "propagation_chain_coverage": 0.0,
            "propagation_chain_coverage_applicable": propagation_chain_coverage_applicable,
        }
        with REWARD_PATH.open("w") as f:
            json.dump(result, f)
        _echo_answer(answer, recovered_from_transcript)
        print(f"ERROR: ground_truth.yaml has no groups[] entry with root_cause: true: {result}")
        return 0

    answer_root_cause = answer["root_cause"]
    answer_kind = answer["kind"]

    expected_kind = root_group.get("kind")
    kind_match = (
        1.0
        if isinstance(expected_kind, str) and answer_kind.lower() == expected_kind.lower()
        else 0.0
    )
    name_match = 1.0 if name_matches(answer_root_cause, group_patterns(root_group)) else 0.0

    expected_namespace = root_group.get("namespace")
    if isinstance(expected_namespace, str) and expected_namespace.strip():
        namespace_applicable = 1.0
        answer_namespace = answer.get("namespace")
        namespace_match = (
            1.0
            if isinstance(answer_namespace, str)
            and answer_namespace.strip().lower() == expected_namespace.strip().lower()
            else 0.0
        )
    else:
        # Vacuous pass -- no namespace to check (e.g. a Namespace-kind root
        # cause has no parent namespace to report). namespace_applicable
        # distinguishes this from a genuine match in aggregate stats.
        namespace_applicable = 0.0
        namespace_match = 1.0

    base_reward = 1.0 if (kind_match == 1.0 and name_match == 1.0 and namespace_match == 1.0) else 0.0
    # Recovery is a diagnostic safety net, not a free pass: a correct answer
    # that only ever existed in chat is still missing the required file, so
    # it's capped at half credit rather than scored the same as a proper
    # answer.json (0 stays 0 -- nothing to halve for a wrong answer).
    reward = base_reward * 0.5 if recovered_from_transcript else base_reward
    chain_proximity, chain_proximity_applicable = compute_chain_proximity(answer_root_cause, ground_truth)
    reasoning_present = compute_reasoning_present(answer)
    propagation_chain_coverage, propagation_chain_coverage_applicable = compute_propagation_chain_coverage(
        answer, ground_truth
    )

    result = {
        "reward": reward,
        "kind_match": kind_match,
        "name_match": name_match,
        "namespace_match": namespace_match,
        "namespace_applicable": namespace_applicable,
        "answer_format_valid": 1.0 if answer_format_valid else 0.0,
        "answer_recovered_from_transcript": 1.0 if recovered_from_transcript else 0.0,
        "chain_proximity": chain_proximity,
        "chain_proximity_applicable": chain_proximity_applicable,
        "reasoning_present": reasoning_present,
        "propagation_chain_coverage": propagation_chain_coverage,
        "propagation_chain_coverage_applicable": propagation_chain_coverage_applicable,
    }
    with REWARD_PATH.open("w") as f:
        json.dump(result, f)
    _echo_answer(answer, recovered_from_transcript)
    print(f"graded: {result}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
