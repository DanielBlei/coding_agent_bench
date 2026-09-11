#!/usr/bin/env python3
"""
Grader for itbench-aa tasks with root_causes array support.

Reads the agent's /workspace/answer.json, compares it against
tests/ground_truth.yaml (copied in at task-generation time, grader-only --
never present under environment/), and writes a multi-metric reward to
/logs/verifier/reward.json.

NEW SCHEMA (v1.0):
  {
    "schema_version": "1.0",            # soft-HARD: logged if missing, default assumed
    "root_causes": [                     # HARD: non-empty list required for format_ok
      {"name": str, "kind": str, "namespace": str}
    ],
    "reasoning": str,                    # LENIENT: presence checked, content scored by LLM
    "propagation_chain": [str],         # LENIENT: checked for edges but not required
    "recommended_actions": [str]        # NONE: captured but never scored
  }

ENFORCEMENT:
- schema_version: if present and != "1.0" -> format_ok=False. If missing, log but proceed.
- root_causes: must be non-empty list -> format_ok=False if absent, empty, or non-list.
- name/kind/namespace: STRICT matching via name_matches_strict + case-insensitive kind.
- reasoning: LENIENT. Presence = whether at least one token resolves to a GT group.
- propagation_chain: LENIENT. Scored on edges covered, not entry count.
- recommended_actions: NONE. Captured in reward but never scored.

SCORING CHANGES FROM OLD:
- headline reward: now recall-gated precision (kind_match AND name_match AND
  namespace_match AND all_gt_covered AND not_recovered).
- All metrics (kind_match, name_match, namespace_match) now use "best per-entity"
  (highest among submitted root_causes).
- chain metrics: replaced chain_proximity with full edge-coverage chain_resolution_rate.
- recovered answers: still cap reward to 0.0 (was 0.5) -- no partial credit for
  transcript recovery on new schema.
- inapplicable metrics: set to None when N/A (e.g., namespace_match when no namespace).
"""

import json
import re
import sys
from pathlib import Path

from matching import (
    WORKLOAD_KINDS,
    derive_matching_literal,
    find_root_cause_group,
    group_patterns,
    name_matches_root_identity,
    name_matches_strict,
    normalize_ground_truth,
    resolve_chain_element,
)

ANSWER_PATH = Path("/workspace/answer.json")
GROUND_TRUTH_PATH = Path(__file__).resolve().parent / "ground_truth.yaml"
REWARD_DIR = Path("/logs/verifier")
REWARD_PATH = REWARD_DIR / "reward.json"
AGENT_LOGS_DIR = Path("/logs/agent")


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
            if not isinstance(entry, dict):
                continue
            # pi / Anthropic-style: {"message": {"role": "assistant", "content": ...}}
            message = entry.get("message")
            if isinstance(message, dict) and message.get("role") == "assistant":
                content = message.get("content")
                if isinstance(content, str):
                    yield mtime, content
                elif isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str):
                            yield mtime, block["text"]
                continue
            # opencode-style event: {"type": "text"|"reasoning", "part": {"text": ...}}
            part = entry.get("part")
            if isinstance(part, dict) and part.get("type") in ("text", "reasoning") and isinstance(part.get("text"), str):
                yield mtime, part["text"]


def recover_answer_from_transcript() -> dict | None:
    """Last-resort recovery: scan the agent's own transcript (most recent
    text first) for a root_causes or root_cause JSON blob it stated but never
    wrote to /workspace/answer.json. Supports both new schema (root_causes array,
    schema_version) and legacy schema (root_cause string). Normalizes any found
    answer to the new schema shape with root_causes array. Never raises --
    returns None on any failure or if nothing matches.
    """
    try:
        candidates = sorted(_iter_transcript_texts(AGENT_LOGS_DIR), key=lambda pair: pair[0])
    except Exception:
        return None

    # Legacy regex: flat object with root_cause/kind (no nested braces)
    legacy_schema_re = re.compile(
        r'\{[^{}]*"root_cause"[^{}]*"kind"[^{}]*\}|\{[^{}]*"kind"[^{}]*"root_cause"[^{}]*\}'
    )

    def _extract_json_objects(text: str) -> list[str]:
        """Extract all balanced-brace substrings from text (handles nesting)."""
        results = []
        i = 0
        n = len(text)
        while i < n:
            if text[i] == '{':
                depth = 0
                for j in range(i, n):
                    c = text[j]
                    if c == '{':
                        depth += 1
                    elif c == '}':
                        depth -= 1
                        if depth == 0:
                            results.append(text[i:j + 1])
                            i = j + 1
                            break
                else:
                    break
            else:
                i += 1
        return results

    for _, text in reversed(candidates):
        # Try new schema first (supports nested objects like root_causes array)
        for blob in reversed(_extract_json_objects(text)):
            try:
                candidate = json.loads(blob)
            except json.JSONDecodeError:
                continue
            if not isinstance(candidate, dict):
                continue

            # Check for new schema (root_causes array)
            root_causes = candidate.get("root_causes")
            if isinstance(root_causes, list) and root_causes:
                # Valid new schema found
                recovered = {
                    "schema_version": candidate.get("schema_version", "1.0"),
                    "root_causes": root_causes,
                }
                if isinstance(candidate.get("reasoning"), str):
                    recovered["reasoning"] = candidate["reasoning"]
                if isinstance(candidate.get("propagation_chain"), list):
                    recovered["propagation_chain"] = candidate["propagation_chain"]
                if isinstance(candidate.get("recommended_actions"), list):
                    recovered["recommended_actions"] = candidate["recommended_actions"]
                return recovered

        # Fall back to legacy schema (root_cause string)
        for blob in reversed(legacy_schema_re.findall(text)):
            try:
                candidate = json.loads(blob)
            except json.JSONDecodeError:
                continue
            if not isinstance(candidate, dict):
                continue

            root_cause = candidate.get("root_cause")
            kind = candidate.get("kind")
            if isinstance(root_cause, str) and isinstance(kind, str):
                # Convert legacy to new schema
                recovered = {
                    "schema_version": "1.0",
                    "root_causes": [
                        {
                            "name": root_cause,
                            "kind": kind,
                        }
                    ]
                }
                if isinstance(candidate.get("namespace"), str):
                    recovered["root_causes"][0]["namespace"] = candidate["namespace"]
                if isinstance(candidate.get("reasoning"), str):
                    recovered["reasoning"] = candidate["reasoning"]
                if isinstance(candidate.get("propagation_chain"), list):
                    recovered["propagation_chain"] = candidate["propagation_chain"]
                return recovered

    return None


def load_answer() -> tuple[dict | None, bool, bool]:
    """Returns (answer_dict_or_None, format_ok, recovered_from_transcript).

    Reads /workspace/answer.json and validates new schema:
    - schema_version: if present and != "1.0" -> format_ok=False. If missing, log but default.
    - root_causes: must be non-empty list -> format_ok=False if absent, empty, or non-list.

    If answer.json is missing/malformed, tries transcript recovery (always converts
    to new schema shape). Returns (answer, format_ok, recovered) where format_ok is
    True only for real answer.json with valid new schema; recovered is True when
    answer came from transcript instead of disk.
    """
    if ANSWER_PATH.is_file():
        try:
            with ANSWER_PATH.open("r") as f:
                answer = json.load(f)
        except (json.JSONDecodeError, OSError):
            answer = None

        if isinstance(answer, dict):
            # Check schema_version
            schema_version = answer.get("schema_version", "1.0")
            if isinstance(schema_version, str) and schema_version != "1.0":
                print(f"WARNING: schema_version '{schema_version}' != '1.0'", file=sys.stderr)
                format_ok = False
            elif not isinstance(schema_version, str) and "schema_version" in answer:
                format_ok = False
            else:
                # Check root_causes
                root_causes = answer.get("root_causes")
                if isinstance(root_causes, list) and root_causes:
                    return answer, True, False
                else:
                    print(f"WARNING: root_causes missing, empty, or non-list", file=sys.stderr)
                    format_ok = False

    # Try transcript recovery
    recovered = recover_answer_from_transcript()
    if recovered is not None:
        return recovered, False, True

    return None, False, False


def _build_alias_map(ground_truth: dict) -> dict[str, list[str]]:
    """Build {group_id: [sibling_group_id, ...]} from ground_truth["aliases"] block.

    Each alias class is a list of group IDs that are aliases of each other.
    For each group_id, map it to all its siblings (same alias class, different id).
    """
    alias_map = {}
    for alias_class in (ground_truth.get("aliases") or []):
        if not isinstance(alias_class, list):
            continue
        for gid in alias_class:
            if isinstance(gid, str):
                alias_map[gid] = [sid for sid in alias_class if sid != gid and isinstance(sid, str)]
    return alias_map


def _compute_ns_applicable(root_group: dict | None) -> float:
    """Helper: return 1.0 if root_group has a non-empty namespace field, else 0.0."""
    if not root_group:
        return 0.0
    ns = root_group.get("namespace")
    return 1.0 if isinstance(ns, str) and ns.strip() else 0.0


def _match_submitted_entity(
    name: str,
    kind: str,
    root_group: dict | None,
    groups_by_id: dict[str, dict],
    alias_map: dict[str, list[str]]
) -> tuple[dict | None, float, float, float | None, float]:
    """Try to match a submitted entity against ground truth groups.

    Returns (matched_group, name_match, kind_match, namespace_match, namespace_applicable).

    Matching rules (first match wins):
    1. Exact same kind: name must strict-fullmatch the root group's patterns
       (unchanged legacy behaviour).
    2. Alias siblings (same alias class): kind matches a sibling's kind AND name
       fullmatches that sibling's filter (unchanged legacy behaviour -- this is
       what lets a legitimately differently-named Service, e.g. `root-svc` for
       pod `root-pod-*`, claim the Pod root).
    3. Workload-equivalence (NEW): when both the submitted and root kind are in
       WORKLOAD_KINDS (Pod/ReplicaSet/Deployment/StatefulSet/DaemonSet/Service),
       accept the root iff the submitted name identifies the ROOT workload itself
       via name_matches_root_identity (strict or suffix-relaxed anchored match of
       the root's OWN patterns). This is what lets a correct governing-controller
       answer (`load-generator`/Deployment) or the bare workload name claim a
       Pod-modelled root, without a sibling filter alone ever claiming the root.

    Namespace matching (only if a group matched):
      - matched_group has a non-empty namespace -> namespace_applicable=1.0
        (namespace_match filled in by caller).
      - else -> namespace_applicable=0.0, namespace_match=None.
    If no group matched: return (None, 0.0, 0.0, None, namespace_applicable from root_group).
    """
    if not root_group:
        return None, 0.0, 0.0, None, 0.0

    root_kind = root_group.get("kind")
    root_kind_l = root_kind.lower() if isinstance(root_kind, str) else ""
    sub_kind_l = kind.lower()
    root_patterns = group_patterns(root_group)

    def _matched(group: dict) -> tuple[dict, float, float, float | None, float]:
        expected_ns = group.get("namespace")
        ns_app = 1.0 if isinstance(expected_ns, str) and expected_ns.strip() else 0.0
        return group, 1.0, 1.0, None, ns_app

    # 1. Exact same kind, strict fullmatch on the root's own patterns.
    if root_kind_l and sub_kind_l == root_kind_l and name_matches_strict(name, root_patterns):
        return _matched(root_group)

    # 2. Alias siblings (legacy): kind + sibling's own filter.
    root_id = root_group.get("id")
    if isinstance(root_id, str):
        for sibling_id in alias_map.get(root_id, []):
            sibling_group = groups_by_id.get(sibling_id)
            if not sibling_group:
                continue
            sibling_kind = sibling_group.get("kind")
            if isinstance(sibling_kind, str) and sub_kind_l == sibling_kind.lower():
                if name_matches_strict(name, group_patterns(sibling_group)):
                    return _matched(sibling_group)

    # 3. Workload equivalence (controller <-> Pod <-> Service), gated on the
    # submitted name matching the ROOT workload identity so a symptom entity can
    # never reach the root through this path.
    if (
        root_kind_l in WORKLOAD_KINDS
        and sub_kind_l in WORKLOAD_KINDS
        and name_matches_root_identity(name, root_patterns)
    ):
        return _matched(root_group)

    # No group matched
    return None, 0.0, 0.0, None, _compute_ns_applicable(root_group)


def _compute_chain_metrics(answer: dict | None, ground_truth: dict, groups_by_id: dict[str, dict]) -> dict:
    """Compute chain metrics: chain_applicable, chain_head_correct, propagation_edge_coverage, chain_resolution_rate.

    Returns dict with those four keys.

    chain_applicable: 1.0 if ground truth has propagations, else 0.0.
    If not applicable, other metrics are None.

    If applicable but answer has no propagation_chain: all metrics except chain_applicable are 0.0.

    Otherwise: compute chain_head_correct (first resolved id is root),
    propagation_edge_coverage (unique GT edges covered by consecutive resolved pairs),
    and chain_resolution_rate (fraction of entries that resolve).
    """
    propagations = [p for p in (ground_truth.get("propagations") or []) if isinstance(p, dict)]
    chain_applicable = 1.0 if propagations else 0.0

    if not chain_applicable:
        return {
            "chain_applicable": 0.0,
            "chain_head_correct": None,
            "propagation_edge_coverage": None,
            "chain_resolution_rate": None,
        }

    # Extract propagation_chain entries from answer
    chain_entries = answer.get("propagation_chain") if isinstance(answer, dict) else None
    if not isinstance(chain_entries, list):
        chain_entries = []
    entries = [e for e in chain_entries if isinstance(e, str) and e.strip()]

    if not entries:
        return {
            "chain_applicable": 1.0,
            "chain_head_correct": 0.0,
            "propagation_edge_coverage": 0.0,
            "chain_resolution_rate": 0.0,
        }

    # Resolve each entry to a group id (root-aware: the bare workload name /
    # owning controller resolves to the Pod root group, while a victim sharing
    # an alias class cannot resolve to the root head).
    root_group = find_root_cause_group(ground_truth)
    root_id = root_group.get("id") if root_group else None
    resolved_ids = [resolve_chain_element(e, groups_by_id, root_group) for e in entries]
    resolution_rate = sum(1 for r in resolved_ids if r is not None) / len(resolved_ids) if resolved_ids else 0.0

    # chain_head_correct: first resolved element should map to root_cause group
    first_resolved = next((r for r in resolved_ids if r is not None), None)
    chain_head_correct = 1.0 if first_resolved and first_resolved == root_id else 0.0

    # propagation_edge_coverage: unique (source, target) GT edges covered by consecutive resolved pairs
    seen_edges = set()
    gt_edges = []
    for p in propagations:
        src, tgt = p.get("source"), p.get("target")
        if isinstance(src, str) and isinstance(tgt, str) and (src, tgt) not in seen_edges:
            gt_edges.append((src, tgt))
            seen_edges.add((src, tgt))

    # Workload canonicalization: resolve_chain_element step 0 maps every element
    # identifying the root workload (bare name, controller, hash-suffixed Pod) to
    # the root group id, so a GT hop *within* one workload (e.g. the root Pod ->
    # its own Service) would otherwise be unmatchable by name (that regression
    # dropped the scenario-1 oracle's edge coverage from 1.0 to 0.0). Collapse
    # both sides of every edge -- GT and resolved chain -- to the root id when the
    # group names the SAME workload identity as the root, using exactly the
    # gate as _match_submitted_entity rule 3 (both kinds in WORKLOAD_KINDS and
    # the group's own derived literal satisfying the root's anchored identity).
    def _workload_canon(gid):
        if gid is None or gid == root_id or root_group is None:
            return gid
        root_kind = root_group.get("kind")
        group = groups_by_id.get(gid)
        if not isinstance(root_kind, str) or not isinstance(group, dict):
            return gid
        group_kind = group.get("kind")
        if root_kind.lower() not in WORKLOAD_KINDS or not isinstance(group_kind, str) or group_kind.lower() not in WORKLOAD_KINDS:
            return gid
        root_patterns = group_patterns(root_group)
        literal = derive_matching_literal(group, group_patterns(group), strict=True)
        if literal is not None and name_matches_root_identity(literal, root_patterns):
            return root_id
        return gid

    canon_resolved = [_workload_canon(r) for r in resolved_ids]
    canon_gt_edges = []
    canon_seen = set()
    for (src, tgt) in gt_edges:
        cpair = (_workload_canon(src), _workload_canon(tgt))
        if cpair not in canon_seen:
            canon_gt_edges.append(cpair)
            canon_seen.add(cpair)

    covered = sum(
        1 for (src, tgt) in canon_gt_edges
        if any(
            canon_resolved[i] == src and canon_resolved[i + 1] == tgt
            for i in range(len(canon_resolved) - 1)
            if canon_resolved[i] is not None and canon_resolved[i + 1] is not None
        )
    )
    edge_coverage = covered / len(canon_gt_edges) if canon_gt_edges else 0.0

    return {
        "chain_applicable": 1.0,
        "chain_head_correct": chain_head_correct,
        "propagation_edge_coverage": edge_coverage,
        "chain_resolution_rate": resolution_rate,
    }


def _compute_reasoning_present(answer: dict | None, groups_by_id: dict[str, dict]) -> float:
    """Return 1.0 if any token in the reasoning resolves to a GT group via resolve_chain_element().
    Else 0.0.
    """
    reasoning = answer.get("reasoning") if isinstance(answer, dict) else None
    if not isinstance(reasoning, str):
        return 0.0
    tokens = re.findall(r'[\w][\w\-]*', reasoning)
    for token in tokens:
        if resolve_chain_element(token, groups_by_id) is not None:
            return 1.0
    return 0.0


def _count_turns(agent_dir: Path) -> int:
    """Count assistant turns across /logs/agent transcripts. Returns 0 if the dir
    doesn't exist.

    Two transcript schemas are supported:
    - pi / Anthropic-style JSONL: the canonical session log has one line per
      message; one turn per line with {"message": {"role": "assistant"}}. Only
      counted from `.jsonl` -- the archived `.txt` event dump repeats each
      assistant message across message_start/message_end/turn_end events (and
      carries no message id to dedupe on), so counting it would 3x-inflate.
    - opencode event schema (either extension): no role field; assistant work is
      emitted as {"type": ..., "part": {"messageID": ...}} events sharing the
      message's id. Counted as distinct assistant messageIDs, deduped across all
      files, matching the pi one-turn-per-message semantics.
    """
    if not agent_dir.is_dir():
        return 0
    pi_count = 0
    opencode_message_ids: set[str] = set()
    for path in agent_dir.rglob("*"):
        if not path.is_file() or path.suffix not in (".jsonl", ".txt"):
            continue
        try:
            lines = path.read_text(errors="ignore").splitlines()
        except OSError:
            continue
        is_jsonl = path.suffix == ".jsonl"
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(entry, dict):
                continue
            part = entry.get("part")
            if isinstance(part, dict) and isinstance(part.get("messageID"), str):
                opencode_message_ids.add(part["messageID"])
                continue
            # pi-style message-per-line: only trustworthy from canonical .jsonl.
            if is_jsonl:
                msg = entry.get("message")
                if isinstance(msg, dict) and msg.get("role") == "assistant":
                    pi_count += 1
    return pi_count + len(opencode_message_ids)


def _echo_answer(answer: dict | None, recovered_from_transcript: bool) -> None:
    """Surface the agent's full raw answer (every key) to verifier stdout
    (test-stdout.txt) for qualitative review -- answer.json itself never leaves
    the container, so this is the only place a human can see what the agent
    actually produced.

    If /workspace/answer.json was missing/malformed, `answer` is instead the
    best-effort fallback recovered from the agent's own transcript -- say so
    explicitly rather than silently printing it as if it were the real file.
    """
    if answer is None:
        print("agent answer.json: NOT FOUND on disk, and no recoverable answer found in transcript.\n")
    elif recovered_from_transcript:
        print(
            "agent answer.json: NOT FOUND on disk -- showing the best-effort "
            "fallback recovered from the agent's transcript instead:\n"
            f"{json.dumps(answer, indent=2)}\n"
        )
    else:
        print(f"agent answer.json:\n{json.dumps(answer, indent=2)}\n")


def main() -> int:
    import yaml

    REWARD_DIR.mkdir(parents=True, exist_ok=True)

    answer, answer_format_valid_bool, recovered_from_transcript = load_answer()
    answer_format_valid = 1.0 if answer_format_valid_bool else 0.0
    turn_count = _count_turns(AGENT_LOGS_DIR)

    # Load ground truth
    try:
        with GROUND_TRUTH_PATH.open("r") as f:
            ground_truth = normalize_ground_truth(yaml.safe_load(f))
    except Exception as e:
        print(f"ERROR: failed to load ground_truth.yaml: {e}", file=sys.stderr)
        # Emit minimal result
        result = {
            "reward": 0.0,
            "answer_format_valid": answer_format_valid,
            "turn_count": turn_count,
        }
        REWARD_PATH.write_text(json.dumps(result))
        return 0

    root_group = find_root_cause_group(ground_truth)
    gt_groups_rc = [
        g for g in (ground_truth.get("groups") or [])
        if isinstance(g, dict) and g.get("root_cause") is True
    ]
    groups_by_id = {
        g["id"]: g for g in (ground_truth.get("groups") or [])
        if isinstance(g, dict) and isinstance(g.get("id"), str)
    }
    alias_map = _build_alias_map(ground_truth)

    # No answer at all
    if answer is None:
        # Compute applicability from GT for honest reporting
        ns_applicable = _compute_ns_applicable(root_group)
        propagations = [p for p in (ground_truth.get("propagations") or []) if isinstance(p, dict)]
        chain_applicable = 1.0 if propagations else 0.0
        result = {
            "reward": 0.0,
            "answer_format_valid": 0.0,
            "answer_recovered_from_transcript": 0.0,
            "name_match": 0.0,
            "kind_match": 0.0,
            "namespace_match": 0.0,
            "namespace_applicable": ns_applicable,
            "reasoning_present": 0.0,
            "chain_applicable": chain_applicable,
            "chain_head_correct": 0.0,
            "propagation_edge_coverage": 0.0,
            "chain_resolution_rate": 0.0,
            "submitted_entity_count": 0.0,
            "turn_count": float(turn_count),
        }
        REWARD_PATH.write_text(json.dumps(result))
        _echo_answer(None, False)
        return 0

    # Score entities
    sub_entities = [e for e in (answer.get("root_causes") or []) if isinstance(e, dict)]
    submitted_entity_count = len(sub_entities)

    best_name_match = 0.0
    best_kind_match = 0.0
    best_namespace_match = None
    best_namespace_applicable = _compute_ns_applicable(root_group)

    # Recall-gated precision: track which GT groups are covered
    claimed_gt: set[str] = set()
    matched_count = 0

    for entry in sub_entities:
        name = entry.get("name", "") if isinstance(entry.get("name"), str) else ""
        kind = entry.get("kind", "") if isinstance(entry.get("kind"), str) else ""
        submitted_ns = entry.get("namespace")

        # Try each unclaimed GT root-cause group in order; claim the first full match
        entry_claimed = False
        for candidate_gt in gt_groups_rc:
            cand_id = candidate_gt.get("id")
            if cand_id in claimed_gt:
                continue

            matched_group, nm, km, _, ns_app = _match_submitted_entity(
                name, kind, candidate_gt, groups_by_id, alias_map
            )

            # Compute namespace_match when applicable
            ns_m = None
            if ns_app == 1.0 and matched_group:
                expected_ns = matched_group.get("namespace")
                if isinstance(expected_ns, str) and expected_ns.strip():
                    ns_m = (
                        1.0
                        if isinstance(submitted_ns, str)
                        and submitted_ns.strip().lower() == expected_ns.strip().lower()
                        else 0.0
                    )

            # Update session-level best diagnostics regardless of outcome
            if nm > best_name_match:
                best_name_match = nm
            if km > best_kind_match:
                best_kind_match = km
            if ns_app:
                best_namespace_applicable = ns_app
            if ns_m is not None and (best_namespace_match is None or ns_m > best_namespace_match):
                best_namespace_match = ns_m

            # Full match requires namespace when GT has one
            if matched_group and not (ns_app == 1.0 and ns_m == 0.0):
                claimed_gt.add(cand_id)
                matched_count += 1
                entry_claimed = True
                break

        # If no candidate matched at all, still capture diagnostics from root_group
        if not entry_claimed and root_group and root_group.get("id") not in claimed_gt:
            _, nm, km, _, ns_app = _match_submitted_entity(
                name, kind, root_group, groups_by_id, alias_map
            )
            if nm > best_name_match:
                best_name_match = nm
            if km > best_kind_match:
                best_kind_match = km

    # Recall-gated precision: must cover all GT root-cause groups
    gt_count = len(gt_groups_rc)
    all_gt_covered = matched_count >= gt_count and gt_count > 0

    # Base reward: precision (matched / submitted) only if all GT covered
    if not sub_entities or not all_gt_covered:
        base_reward = 0.0
    else:
        base_reward = matched_count / submitted_entity_count

    # No partial credit for transcript recovery on new schema
    reward = 0.0 if recovered_from_transcript else base_reward

    # Chain and reasoning metrics
    chain_metrics = _compute_chain_metrics(answer, ground_truth, groups_by_id)
    reasoning_present = _compute_reasoning_present(answer, groups_by_id)

    result = {
        "reward": reward,
        "answer_format_valid": answer_format_valid,
        "answer_recovered_from_transcript": 1.0 if recovered_from_transcript else 0.0,
        "name_match": best_name_match,
        "kind_match": best_kind_match,
        # namespace_match: 0.0 when applicable but missed, OR when inapplicable
        # (namespace_applicable==0.0 means nothing was checked; emit 0.0 not vacuous 1.0).
        # Conditioned rate = mean(namespace_match) / mean(namespace_applicable).
        "namespace_match": best_namespace_match if best_namespace_match is not None else 0.0,
        "namespace_applicable": best_namespace_applicable,
        "reasoning_present": reasoning_present,
        "chain_applicable": chain_metrics["chain_applicable"],
        # chain metrics: 0.0 when applicable but missing/empty chain.
        # chain_applicable==0.0 signals "no propagations to check" (not a failure).
        "chain_head_correct": chain_metrics["chain_head_correct"] if chain_metrics["chain_head_correct"] is not None else 0.0,
        "propagation_edge_coverage": chain_metrics["propagation_edge_coverage"] if chain_metrics["propagation_edge_coverage"] is not None else 0.0,
        "chain_resolution_rate": chain_metrics["chain_resolution_rate"] if chain_metrics["chain_resolution_rate"] is not None else 0.0,
        "submitted_entity_count": float(submitted_entity_count),
        "turn_count": float(turn_count),
    }
    # NONE-class fields: printed for qualitative review, never in reward.json
    rec_actions = answer.get("recommended_actions")
    if rec_actions:
        print(f"recommended_actions: {json.dumps(rec_actions)}")

    REWARD_PATH.write_text(json.dumps(result))
    _echo_answer(answer, recovered_from_transcript)
    print(f"graded: {result}")
    return 0


if __name__ == "__main__":
    sys.exit(main())