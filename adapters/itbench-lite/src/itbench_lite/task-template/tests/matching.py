"""
Shared root-cause matching logic for itbench-lite tasks.

Used by both `adapter.py` (to build an oracle answer that is provably
gradeable, instead of assuming the ground truth's synthetic group `id`
happens to satisfy the grading rule) and the generated `tests/grade.py` (to
score an agent's answer). Keeping this in one module means the two can never
drift out of sync with each other again -- `adapter.py` imports it directly,
and a copy is placed alongside `grade.py` in every generated task's
`tests/` directory (see `ITBenchLiteAdapter._render_task`).
"""

from __future__ import annotations

import fnmatch
import re


def normalize_ground_truth(ground_truth: dict) -> dict:
    """Some scenarios wrap their content in a Kubernetes-CRD-style envelope
    (apiVersion/kind: GroundTruth/metadata/spec) instead of the flat shape
    (top-level groups/alerts/fault/propagations) used elsewhere -- same
    semantic content, one extra level of nesting under `spec`. Normalize to
    the flat shape so every other function here can stay ignorant of which
    envelope it was loaded from.
    """
    spec = ground_truth.get("spec")
    if isinstance(spec, dict) and "groups" not in ground_truth and "groups" in spec:
        return spec
    return ground_truth


def find_root_cause_group(ground_truth: dict) -> dict | None:
    for group in ground_truth.get("groups", []) or []:
        if group.get("root_cause") is True:
            return group
    return None


def group_patterns(group: dict) -> list:
    """Patterns identifying a group. Prefer its filter[] list; when absent
    (e.g. a ConfigMap root cause that ships only a name), fall back to an
    anchored exact match on name so it is still gradeable."""
    filters = group.get("filter")
    if isinstance(filters, list) and filters:
        return [f for f in filters if isinstance(f, str)]
    name = group.get("name")
    if isinstance(name, str) and name:
        return [f"^{re.escape(name)}$"]
    return []


def name_matches(name: str, patterns: list) -> bool:
    for pattern in patterns:
        try:
            if re.search(pattern, name):
                return True
            # Owned objects (Pods/ReplicaSets) get a random hash suffix that
            # unowned resources (Deployments, Services, ConfigMaps, ...)
            # never do. A "<literal>-.*"/"<literal>-.+" filter written for
            # the suffixed case should also match the bare, unsuffixed
            # literal -- otherwise a factually correct answer naming the
            # real (suffix-less) resource can never pass. Strictly widens
            # matching (only adds the zero-length-suffix case), so this
            # can't turn a previously-failing match into a false positive
            # relative to what the unrelaxed pattern already allowed.
            relaxed = re.sub(r"-(?:\.\*|\.\+)$", "", pattern)
            if relaxed != pattern and re.search(relaxed, name):
                return True
            # This dataset is built on the OTel Demo ("Astronomy Shop") app,
            # whose services have two names in play: the OTel service.name
            # used in traces/descriptive fault text (e.g. "adservice"), and
            # the actual k8s object short name (e.g. "ad") used by Chaos
            # Mesh CR names and most filters. A pattern containing a bare
            # "<word>service" token should also match that word as a whole
            # hyphen/underscore-delimited name segment -- bounded, not a raw
            # substring, so this can't spuriously match an unrelated name
            # that merely contains the same letters (e.g. "ad" inside
            # "load-generator").
            service_word = re.search(r"([a-zA-Z][a-zA-Z0-9]*)service\b", pattern, re.IGNORECASE)
            if service_word:
                token_pattern = rf"(?:^|[-_/]){re.escape(service_word.group(1))}(?:[-_/]|$)"
                if re.search(token_pattern, name, re.IGNORECASE):
                    return True
        except re.error:
            # Not valid Python regex -- this dataset's convention for such
            # filters is a "<namespace-glob>.<name-glob>" selector (e.g.
            # "*.*" = any object of this kind, in this namespace). The
            # agent's answer schema has no namespace field to check against,
            # so we match only the name-glob segment; kind_match still gates
            # on the object's kind.
            name_glob = pattern.rsplit(".", 1)[-1] if "." in pattern else pattern
            if fnmatch.fnmatch(name, name_glob):
                return True
    return False


def _synthesize_literal(pattern: str) -> str | None:
    """Best-effort: derive a concrete literal from a single pattern's own
    shape, without any knowledge of real telemetry data. Only ever used as a
    candidate that is then self-verified via name_matches() before being
    trusted -- never returned on faith.
    """
    try:
        re.compile(pattern)
    except re.error:
        # Glob-style selector (see name_matches) -- replace wildcards with a
        # literal filler so the result is a concrete, matchable name.
        name_glob = pattern.rsplit(".", 1)[-1] if "." in pattern else pattern
        literal = name_glob.replace("*", "x").replace("?", "x")
        return literal or None

    core = pattern.lstrip("^").rstrip("$")
    core = core.replace(r"\b", "")
    core = re.sub(r"^\.\*", "", core)
    core = re.sub(r"\.\*$", "", core)
    core = re.sub(r"\\(.)", r"\1", core)  # unescape re.escape-style literals
    return core or None


def derive_matching_literal(root_group: dict, patterns: list) -> str | None:
    """Derive a literal string guaranteed (self-verified) to satisfy
    name_matches(literal, patterns) for a root-cause group -- so an oracle
    answer built from it is provably gradeable instead of assuming the
    group's internal synthetic id happens to coincide with the grading
    pattern. Tries the group's own id/name first (often already valid, e.g.
    when a filter is a prefix of the id) before synthesizing a literal from
    the pattern's own regex/glob shape. Returns None only if truly nothing
    can be derived (should not happen for any pattern list produced by
    group_patterns()).
    """
    if not patterns:
        return None
    for candidate in (root_group.get("id"), root_group.get("name")):
        if isinstance(candidate, str) and candidate and name_matches(candidate, patterns):
            return candidate
    for pattern in patterns:
        literal = _synthesize_literal(pattern)
        if literal is not None and name_matches(literal, patterns):
            return literal
    return None
