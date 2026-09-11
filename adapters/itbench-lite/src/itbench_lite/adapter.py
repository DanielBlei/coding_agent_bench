"""
Adapter class that converts ITBench-Lite (sre-domain) scenarios into Harbor tasks.

Dataset shape: file-tree. One item = one ``Scenario-N/`` directory holding a
snapshot of a Kubernetes cluster during a simulated incident (alerts, metrics,
k8s events/objects, otel logs/traces) plus a ``ground_truth.yaml`` answer key
identifying the root-cause entity.

Implements run() to enumerate scenarios under the downloaded HF snapshot and
render one Harbor task per scenario into self.output_dir.
"""

from __future__ import annotations

import importlib.util
import json
import re
import shlex
import shutil
import sys
from pathlib import Path

import yaml

# Repo root, computed the same way main.py computes its DEFAULT_OUTPUT_DIR:
# this file lives at adapters/itbench-lite/src/itbench_lite/adapter.py, so
# parents[4] is the repo root.
REPO_ROOT = Path(__file__).resolve().parents[4]

# Where the raw HF snapshot download lives (decoupled from self.output_dir,
# which points at datasets/itbench-lite/harbor_tasks/ per this repo's
# convention -- see references/harbor-cli.md, "Repo convention vs. upstream").
DATASET_DOWNLOAD_ROOT = REPO_ROOT / "datasets" / "itbench-lite"

# The six agent-visible input subpaths under each Scenario-N/ directory.
# These are exactly what gets bind-mounted into /workspace/ -- never the
# scenario root, which also holds ground_truth.yaml.
INPUT_SUBPATHS = [
    "alerts",
    "metrics",
    "k8s_events_raw.tsv",
    "k8s_objects_raw.tsv",
    "otel_logs_raw.tsv",
    "otel_traces_raw.tsv",
]

SCENARIO_DIR_RE = re.compile(r"^Scenario-(\d+)$")


def _resolve_alerts_mount(scenario_dir: Path) -> tuple[str, str] | None:
    """Returns (source_relpath, target_relpath) for the "alerts" input,
    relative to the scenario dir and to /workspace/ respectively. Normally
    both are just "alerts" (a directory of periodic alert snapshots); one
    upstream scenario instead ships a single loose alerts_*.json file at the
    scenario root (a one-shot query result, not a snapshot series) -- mount
    that file into /workspace/alerts/<its own filename> so the agent still
    finds it via the documented `alerts/*.json` glob. Returns None if
    neither shape is present (caller treats this as a missing input).
    """
    if (scenario_dir / "alerts").is_dir():
        return "alerts", "alerts"
    matches = sorted(scenario_dir.glob("alerts*.json"))
    if len(matches) == 1:
        return matches[0].name, f"alerts/{matches[0].name}"
    return None


def _input_path_exists(scenario_dir: Path, subpath: str) -> bool:
    if subpath == "alerts":
        return _resolve_alerts_mount(scenario_dir) is not None
    return (scenario_dir / subpath).exists()

TASK_TEMPLATE_DIR = Path(__file__).resolve().parent / "task-template"

# matching.py is the single source of truth for root-cause matching, shared
# by both this adapter (to build a provably-gradeable oracle answer) and the
# generated tests/grade.py (to score an agent's answer). It lives inside
# task-template/tests/ -- the same copy that copytree ships into every
# generated task -- rather than as a separate package module, so there is
# only ever one file to keep in sync.
_MATCHING_PATH = TASK_TEMPLATE_DIR / "tests" / "matching.py"
_matching_spec = importlib.util.spec_from_file_location("itbench_lite_matching", _MATCHING_PATH)
matching = importlib.util.module_from_spec(_matching_spec)
_matching_spec.loader.exec_module(matching)


class ScenarioItem:
    """One enumerated, validated Scenario-N/ directory."""

    def __init__(self, scenario_dir: Path, number: int):
        self.scenario_dir = scenario_dir
        self.number = number
        self.task_id = f"scenario-{number}"
        self.ground_truth: dict | None = None
        self.root_cause_group: dict | None = None


def _resolve_snapshot_root() -> Path:
    """
    Resolve the single snapshot-version directory under
    datasets/itbench-lite/snapshots/sre/*/. Fail fast if there isn't exactly
    one match -- this directory's name is a content-addressed HF snapshot
    hash we never hardcode.
    """
    matches = sorted(
        p for p in (DATASET_DOWNLOAD_ROOT / "snapshots" / "sre").glob("*") if p.is_dir()
    )
    if len(matches) != 1:
        raise RuntimeError(
            "expected exactly one snapshot-version dir under "
            f"{DATASET_DOWNLOAD_ROOT / 'snapshots' / 'sre'}/*, found {len(matches)}: "
            f"{matches}"
        )
    return matches[0]


def _validate_scenario(scenario_dir: Path) -> tuple[dict, dict] | None:
    """
    Validate a single Scenario-N/ directory per the plan's irregular-item
    handling: ground_truth.yaml present + parses, at least one groups[] has
    root_cause: true, all six input paths exist. Returns
    (ground_truth, root_cause_group) on success, None on failure (caller
    skips-and-logs).
    """
    gt_path = scenario_dir / "ground_truth.yaml"
    if not gt_path.is_file():
        print(f"WARNING: skipping {scenario_dir.name}: missing ground_truth.yaml", file=sys.stderr)
        return None

    try:
        with gt_path.open("r") as f:
            ground_truth = yaml.safe_load(f)
    except yaml.YAMLError as e:
        print(f"WARNING: skipping {scenario_dir.name}: ground_truth.yaml failed to parse: {e}", file=sys.stderr)
        return None

    if not isinstance(ground_truth, dict):
        print(f"WARNING: skipping {scenario_dir.name}: ground_truth.yaml did not parse to a mapping", file=sys.stderr)
        return None

    # Some scenarios wrap their content in a CRD-style envelope
    # (apiVersion/kind: GroundTruth/metadata/spec) instead of the flat shape
    # used elsewhere -- normalize once here so everything downstream (this
    # function, _oracle_answer, item.ground_truth) sees the flat shape.
    # tests/ground_truth.yaml is still copied verbatim (see _render_task);
    # grade.py normalizes it the same way at grading time.
    ground_truth = matching.normalize_ground_truth(ground_truth)

    root_cause_group = matching.find_root_cause_group(ground_truth)
    if root_cause_group is None:
        print(f"WARNING: skipping {scenario_dir.name}: no groups[] entry with root_cause: true", file=sys.stderr)
        return None

    if not isinstance(root_cause_group.get("id"), str) or not isinstance(root_cause_group.get("kind"), str):
        print(f"WARNING: skipping {scenario_dir.name}: root-cause group missing string id/kind", file=sys.stderr)
        return None

    # The root-cause group must be gradeable: matching.group_patterns() must
    # yield at least one pattern (filter[] regex/glob list, or a name-based
    # anchor fallback), and a literal answer must actually be derivable from
    # it -- see matching.derive_matching_literal(). This is also what
    # _oracle_answer() relies on to build a provably-gradeable answer.
    patterns = matching.group_patterns(root_cause_group)
    if not patterns:
        print(
            f"WARNING: skipping {scenario_dir.name}: root-cause group has neither filter[] nor name -- ungradeable",
            file=sys.stderr,
        )
        return None
    if matching.derive_matching_literal(root_cause_group, patterns) is None:
        print(
            f"WARNING: skipping {scenario_dir.name}: root-cause group filter[] has no derivable "
            "matching literal -- ungradeable",
            file=sys.stderr,
        )
        return None

    missing = [sub for sub in INPUT_SUBPATHS if not _input_path_exists(scenario_dir, sub)]
    if missing:
        print(f"WARNING: skipping {scenario_dir.name}: missing input path(s): {missing}", file=sys.stderr)
        return None

    return ground_truth, root_cause_group


def _enumerate_items(snapshot_root: Path) -> list[ScenarioItem]:
    items: list[ScenarioItem] = []
    for scenario_dir in sorted(snapshot_root.glob("Scenario-*")):
        if not scenario_dir.is_dir():
            continue
        m = SCENARIO_DIR_RE.match(scenario_dir.name)
        if not m:
            print(f"WARNING: skipping {scenario_dir.name}: does not match Scenario-<N> pattern", file=sys.stderr)
            continue
        number = int(m.group(1))
        validated = _validate_scenario(scenario_dir)
        if validated is None:
            continue
        ground_truth, root_cause_group = validated
        item = ScenarioItem(scenario_dir, number)
        item.ground_truth = ground_truth
        item.root_cause_group = root_cause_group
        items.append(item)
    return items


class ITBenchLiteAdapter:
    def __init__(
        self,
        output_dir: Path,
        limit: int | None = None,
        overwrite: bool = False,
        task_ids: list[str] | None = None,
        **kwargs,
    ):
        """
        Initialize the adapter.

        Args:
            output_dir: directory to write generated Harbor tasks into
                (datasets/itbench-lite/harbor_tasks/).
            limit: maximum number of tasks to generate.
            overwrite: whether to overwrite existing task directories.
            task_ids: if given, only generate these task IDs (e.g. "scenario-1").
            **kwargs: unused, accepted for forward compatibility.
        """
        self.output_dir = Path(output_dir)
        self.limit = limit
        self.overwrite = overwrite
        self.task_ids = task_ids

    def run(self) -> None:
        """
        Enumerate Scenario-N/ directories in the downloaded snapshot, validate
        each, and render one Harbor task per valid scenario into
        self.output_dir.
        """
        snapshot_root = _resolve_snapshot_root()

        items = _enumerate_items(snapshot_root)
        if not items:
            raise RuntimeError(
                f"no valid scenarios found under {snapshot_root} -- every "
                "Scenario-* directory failed validation"
            )
        # Total number of valid scenarios found under snapshot_root, before
        # any --task-ids/--limit filtering -- used to report real coverage
        # in the generated runbook (see _write_runbook), so that text stays
        # accurate however much of the snapshot has actually been fetched.
        total_valid_scenarios = len(items)

        if self.task_ids:
            wanted = set(self.task_ids)
            items = [it for it in items if it.task_id in wanted]
        if self.limit is not None:
            items = items[: self.limit]

        self.output_dir.mkdir(parents=True, exist_ok=True)

        generated = 0
        for item in items:
            dst = self.output_dir / item.task_id
            if dst.exists() and not self.overwrite:
                print(f"skipping {item.task_id}: already exists (use --overwrite to regenerate)")
                continue
            if dst.exists() and self.overwrite:
                shutil.rmtree(dst)
            self._render_task(dst, item, snapshot_root)
            generated += 1
            print(f"generated {item.task_id} -> {dst}")

        # Count task dirs actually present in output_dir (not just ones
        # rendered this run) so the runbook reflects true coverage even
        # across incremental --task-ids/--limit invocations.
        existing_task_count = sum(
            1 for p in self.output_dir.glob("scenario-*") if p.is_dir()
        )
        self._write_runbook(snapshot_root, total_valid_scenarios, existing_task_count)
        print(f"done: generated {generated} task(s) into {self.output_dir}")

    def _render_task(self, dst: Path, item: ScenarioItem, snapshot_root: Path) -> None:
        shutil.copytree(
            TASK_TEMPLATE_DIR,
            dst,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )

        # task.toml
        task_toml_path = dst / "task.toml"
        task_toml = task_toml_path.read_text()
        task_toml = task_toml.replace("{task_id}", item.task_id)
        task_toml_path.write_text(task_toml)

        # instruction.md
        instruction_path = dst / "instruction.md"
        instruction = instruction_path.read_text()
        instruction = instruction.replace("{{PROBLEM_STATEMENT}}", self._instruction_body())
        instruction_path.write_text(instruction)

        # solution/solve.sh -- oracle, rendered with the literal root-cause
        # answer known at generation time: entity + kind + a reasoning blurb
        # and propagation chain derived from the ground truth itself.
        # shlex.quote so nothing in the (possibly third-party) ground-truth
        # strings is shell-interpolated.
        solve_path = dst / "solution" / "solve.sh"
        answer_json = json.dumps(self._oracle_answer(item))
        solve_sh = (
            "#!/bin/sh\n"
            "set -e\n"
            f"echo {shlex.quote(answer_json)} > /workspace/answer.json\n"
        )
        solve_path.write_text(solve_sh)
        solve_path.chmod(0o755)

        # tests/test.sh already ships from task-template as-is (generic
        # grader logic); copy the per-scenario ground_truth.yaml alongside it
        # so it can be parsed at grading time. Grader-only: never lands in
        # environment/ or instruction.md.
        # tests/matching.py already shipped via the copytree above (the
        # single source of truth also used by _oracle_answer() below).
        gt_src = item.scenario_dir / "ground_truth.yaml"
        gt_dst = dst / "tests" / "ground_truth.yaml"
        shutil.copyfile(gt_src, gt_dst)
        test_sh_path = dst / "tests" / "test.sh"
        test_sh_path.chmod(0o755)

        # environment/docker-compose.yaml -- bind-mount the six input
        # subpaths (read-only) from ${DATA_ROOT}/Scenario-<N>/<subpath> to
        # /workspace/<subpath>. Never mount the scenario root itself. Source
        # and target relpaths normally match; "alerts" can differ when the
        # scenario ships a single loose alerts_*.json file instead of an
        # alerts/ directory (see _resolve_alerts_mount).
        compose_template_path = TASK_TEMPLATE_DIR / "environment" / "docker-compose.yaml.tmpl"
        compose_out_path = dst / "environment" / "docker-compose.yaml"
        compose_src = compose_template_path.read_text()
        mounts = []
        for sub in INPUT_SUBPATHS:
            if sub == "alerts":
                source_rel, target_rel = _resolve_alerts_mount(item.scenario_dir)
            else:
                source_rel = target_rel = sub
            mounts.append((source_rel, target_rel))
        volumes = "\n".join(
            self._compose_volume_block(scenario_name=item.scenario_dir.name, source_rel=source_rel, target_rel=target_rel)
            for source_rel, target_rel in mounts
        )
        compose_out = compose_src.replace("__VOLUMES_PLACEHOLDER__", volumes)
        compose_out_path.write_text(compose_out)
        # copytree already copied the .tmpl into dst/environment/; remove it
        # now that the rendered docker-compose.yaml has been written.
        (dst / "environment" / "docker-compose.yaml.tmpl").unlink()

    @staticmethod
    def _oracle_answer(item: ScenarioItem) -> dict:
        """
        Build the full oracle answer (entity + kind + reasoning +
        propagation_chain) from the ground truth. root_cause is derived via
        matching.derive_matching_literal() rather than the group's raw
        (synthetic) id -- the id is only used internally to walk the
        propagation graph, since it is not guaranteed to itself satisfy the
        group's own filter/name matching rule (see matching.py). The
        propagation chain is a walk from the root-cause group id along
        propagations[].source -> .target edges until no further unseen
        target exists, with the first entry swapped for the derived,
        provably-gradeable root_cause so the answer stays self-consistent;
        the reasoning is a deterministic blurb assembled from the fault and
        propagation condition/effect strings. Both exercise the same grader
        paths a real agent's answer would.
        """
        gt = item.ground_truth or {}
        root = item.root_cause_group or {}
        root_id = root.get("id")
        root_kind = root.get("kind")

        patterns = matching.group_patterns(root)
        answer_root_cause = matching.derive_matching_literal(root, patterns)
        if answer_root_cause is None:
            # _validate_scenario() already guarantees this is derivable for
            # any scenario that reaches rendering -- fail loudly rather than
            # silently ship an oracle answer that can't score reward=1.0.
            raise RuntimeError(
                f"{item.task_id}: could not derive a matching root_cause literal from {patterns!r}"
            )
        # Falls back to the derived root_cause literal only for the one
        # Namespace-kind-root-cause case with no parent namespace of its
        # own -- grade.py treats that case as a vacuous pass regardless of
        # what's supplied here, so the exact fallback value doesn't matter.
        answer_namespace = root.get("namespace") or answer_root_cause

        # Breadth-first walk of the full propagation graph from the root
        # cause (using internal ids, so lookups/dedup stay exact) -- several
        # scenarios branch (one root fanning out to multiple independently
        # affected downstream services), and a single linear path would miss
        # every branch but the first, leaving propagation_chain_coverage
        # permanently partial for something entirely within our control.
        # Every visited id is then mapped to a matching literal via the same
        # derive_matching_literal() rule as the root -- so propagation_chain
        # entries actually satisfy their own group's filter/name pattern
        # instead of exposing internal graph ids.
        chain: list[str] = []
        if isinstance(root_id, str):
            edges = [
                (p.get("source"), p.get("target"))
                for p in gt.get("propagations", []) or []
                if isinstance(p, dict)
            ]
            chain = [root_id]
            queue = [root_id]
            while queue:
                current = queue.pop(0)
                for s, t in edges:
                    if s == current and isinstance(t, str) and t not in chain:
                        chain.append(t)
                        queue.append(t)

            groups_by_id = {
                g["id"]: g
                for g in gt.get("groups", []) or []
                if isinstance(g, dict) and isinstance(g.get("id"), str)
            }

            def _display_name(internal_id: str) -> str:
                if internal_id == root_id:
                    return answer_root_cause
                group = groups_by_id.get(internal_id)
                if group is None:
                    return internal_id
                literal = matching.derive_matching_literal(group, matching.group_patterns(group))
                return literal if literal is not None else internal_id

            chain = [_display_name(entry) for entry in chain]

        # Assemble a reasoning blurb from the ground truth narrative. Some
        # scenarios list one fault entry per affected component (root cause
        # plus downstream symptoms, e.g. Scenario-2's frontend/cart/
        # flagd-config trio) -- prefer only the entry that actually
        # identifies the root-cause object when one matches, so the
        # reasoning doesn't read as "the root cause is everything".
        root_name = root.get("name") if isinstance(root.get("name"), str) else None
        all_faults = [f for f in gt.get("fault", []) or [] if isinstance(f, dict)]

        def _is_root_fault(fault: dict) -> bool:
            entity = fault.get("entity", {}) or {}
            return entity.get("group_id") == root_id or entity.get("name") in (root_id, root_name)

        root_faults = [f for f in all_faults if _is_root_fault(f)]
        faults = root_faults or all_faults

        bits: list[str] = []
        for fault in faults:
            entity = fault.get("entity", {}) or {}
            cond = fault.get("condition")
            mech = fault.get("fault_mechanism")
            desc = f"Fault in {entity.get('name', root_id)} ({entity.get('kind', root_kind)})"
            if isinstance(cond, str) and cond and cond.lower() != "to be specified":
                desc += f": {cond}"
            if isinstance(mech, str) and mech and mech.lower() != "custom":
                desc += f" [{mech}]"
            bits.append(desc + ".")
        for p in gt.get("propagations", []) or []:
            if not isinstance(p, dict):
                continue
            effect = p.get("effect")
            if isinstance(effect, str) and effect:
                bits.append(f"{p.get('source')} -> {p.get('target')}: {effect}.")
        reasoning = " ".join(bits) if bits else "See ground truth."

        return {
            "root_cause": answer_root_cause,
            "kind": root_kind,
            "namespace": answer_namespace,
            "reasoning": reasoning,
            "propagation_chain": chain,
        }

    @staticmethod
    def _compose_volume_block(scenario_name: str, source_rel: str, target_rel: str) -> str:
        return (
            "      - type: bind\n"
            f"        source: ${{DATA_ROOT}}/{scenario_name}/{source_rel}\n"
            f"        target: /workspace/{target_rel}\n"
            "        read_only: true"
        )

    @staticmethod
    def _instruction_body() -> str:
        return (
            "# SRE incident investigation\n\n"
            "You are an SRE (Site Reliability Engineer) investigating a live "
            "production incident in a Kubernetes cluster. Alerts have fired, "
            "and telemetry has been captured covering the incident window. "
            "Your job is to identify the **root-cause entity**: the single, "
            "specific Kubernetes object whose failure or misconfiguration "
            "triggered the cascade of symptoms seen in the telemetry -- not "
            "just a symptom, and not a downstream object that was only "
            "affected by the propagated failure.\n\n"
            "## Available data (read-only, under /workspace/)\n\n"
            "- `alerts/` -- per-minute snapshots of firing alerts (JSON, "
            "Prometheus Alertmanager format). Start here to see which "
            "services were affected and when.\n"
            "- `metrics/` -- Prometheus metric samples, one TSV per "
            "pod/service (`pod_<name>_raw.tsv`, `service_<name>_raw.tsv`). "
            "Columns: `metric_name, timestamp, value, pod_name`/"
            "`service_name, namespace, tags` (service files add `bucket_le, "
            "metric_type, status_code` for latency histograms). `tags` is "
            "a Python-style `{'key': 'value'}` string, not JSON -- parse it "
            "accordingly. Values are raw counters/gauges/histogram buckets, "
            "not pre-computed rates.\n"
            "- `k8s_events_raw.tsv` / `k8s_objects_raw.tsv` -- OpenTelemetry "
            "log-record exports, not a flat kubectl-style table: the actual "
            "Kubernetes event/object JSON (with its own `kind`/`name`/"
            "`namespace`/spec) is inside the `Body` column of each row.\n"
            "- `otel_logs_raw.tsv` -- OpenTelemetry logs.\n"
            "- `otel_traces_raw.tsv` -- OpenTelemetry traces.\n\n"
            "These four `*_raw.tsv` files are quoted (fields can contain "
            "embedded quotes and even newlines) and can be large (tens to "
            "hundreds of MB) -- read them with a real TSV/CSV parser rather "
            "than naive line-by-line tools, and prefer targeted queries "
            "over loading a whole file into context. Available in this "
            "environment: `jq` (for the `alerts/*.json` files), `mlr` "
            "(miller -- respects the quoting above; can also convert TSV "
            "rows to JSON to pipe into `jq`), `rg` (ripgrep, for fast "
            "full-text search over the largest files), and Python's "
            "`pandas` (`pandas.read_csv(path, sep=\"\\t\")` also handles "
            "the quoting correctly).\n\n"
            "Correlate across these sources to trace the symptoms back to "
            "the single object that started the cascade.\n\n"
            "## Your answer\n\n"
            "Write your conclusion to `/workspace/answer.json` as a JSON "
            "object with exactly this shape:\n\n"
            "```json\n"
            "{\n"
            '  "root_cause": "<entity-name>",\n'
            '  "kind": "<kind>",\n'
            '  "namespace": "<namespace>",\n'
            '  "reasoning": "<short explanation>",\n'
            '  "propagation_chain": ["<root-entity>", "...", "<symptom-entity>"]\n'
            "}\n"
            "```\n\n"
            "- `root_cause` is the name of the specific object whose own "
            "configuration or state is actually broken -- not an object "
            "that is merely exhibiting symptoms of a failure that "
            "originated elsewhere. If a Pod is unhealthy only because of a "
            "misconfiguration in the Deployment, ConfigMap, or "
            "NetworkPolicy that governs it, the root cause is that "
            "governing object, not the symptomatic Pod; if the Pod itself "
            "is the object that's actually broken (e.g. a fault injected "
            "directly into it), answer with the Pod. Do not answer with a "
            "downstream service or object that merely exhibited the "
            "propagated symptoms. Use the exact name as it appears in the "
            "telemetry (for example, the full pod name as listed in "
            "`k8s_objects_raw.tsv` or `metrics/`, e.g. "
            "`checkout-5dccddf8bb-vh65b`) -- note that not every object "
            "gets a random suffix: Deployments, Services, ConfigMaps, and "
            "similar unowned resources typically keep a stable, "
            "human-chosen name, while their owned Pods/ReplicaSets do "
            "not.\n"
            "- `kind` is the Kubernetes kind of that same object (e.g. "
            "`Pod`, `Service`, `Deployment`, `ConfigMap`). For objects found "
            "via `k8s_objects_raw.tsv`, this is the `kind` field inside "
            "that row's JSON `Body`, not a separate column.\n"
            "- `namespace` is the Kubernetes namespace the root-cause "
            "object lives in (e.g. `otel-demo`). If the root-cause object "
            "is itself a Namespace, repeat its own name here.\n"
            "- `reasoning` is a short explanation (a few sentences) of "
            "**what** is wrong with the root-cause object, **why** it is "
            "the root cause rather than a symptom, and **how** the failure "
            "propagated to produce the observed alerts -- cite the concrete "
            "telemetry evidence (events, logs, metrics, traces) you based "
            "this on.\n"
            "- `propagation_chain` is the ordered path the failure took, "
            "as entity names exactly as they appear in the telemetry, "
            "starting at the root-cause object and ending at the object(s) "
            "that raised the alerts. If the fault is confined to the "
            "root-cause object itself with no observed downstream cascade, "
            "a single-element list containing just the root cause is "
            "valid.\n\n"
            "Example (illustrative only, not real entities in this "
            "scenario):\n\n"
            "```json\n"
            "{\n"
            '  "root_cause": "example-pod-7d5756fbd9-abcde",\n'
            '  "kind": "Pod",\n'
            '  "namespace": "example-namespace",\n'
            '  "reasoning": "The example pod was OOMKilled repeatedly '
            "after a config change (k8s_events_raw.tsv); its 5xx rate "
            "spiked first and only then did the dependent service degrade, "
            "so the pod -- not the service -- is the origin.\",\n"
            '  "propagation_chain": ["example-pod-7d5756fbd9-abcde", '
            '"example-service", "frontend-proxy-6b4d584985-kxvn6"]\n'
            "}\n"
            "```\n\n"
            "**The task is not complete until `/workspace/answer.json` "
            "exists on disk.** Write your answer to this file using a "
            "file-write tool before finishing -- a conclusion stated only "
            "in chat is not a substitute.\n"
        )

    def _write_runbook(
        self, snapshot_root: Path, total_valid_scenarios: int, existing_task_count: int
    ) -> None:
        readme_path = self.output_dir / "README.md"
        readme_path.write_text(
            self._runbook_text(snapshot_root, total_valid_scenarios, existing_task_count)
        )

    @staticmethod
    def _runbook_text(
        snapshot_root: Path, total_valid_scenarios: int, existing_task_count: int
    ) -> str:
        snapshot_root_rel = snapshot_root.relative_to(REPO_ROOT)
        data_root = f'"$(pwd)/{snapshot_root_rel}"'
        # Computed fresh on every run() from the actual snapshot dir and
        # output_dir contents -- do not hardcode a scenario count/name here,
        # it will drift the moment more of the snapshot is fetched or more
        # tasks are generated (see the Overview/Notes sections of the
        # adapter's own README.md for the history of that exact drift).
        if existing_task_count >= total_valid_scenarios:
            coverage_note = (
                f"All {total_valid_scenarios} valid scenario(s) found under "
                f"`{snapshot_root_rel}` have been generated as Harbor tasks."
            )
        else:
            coverage_note = (
                f"{existing_task_count} of {total_valid_scenarios} valid scenario(s) found "
                f"under `{snapshot_root_rel}` have been generated as Harbor tasks so far. Run "
                "without `--task-ids`/`--limit` (add `--overwrite` to also refresh existing "
                "ones) to generate the rest."
            )
        return rf"""# itbench-lite generated tasks

**Source:** [ibm-research/ITBench-Lite](https://huggingface.co/datasets/ibm-research/ITBench-Lite)

**This directory is generated. Do not hand-edit it.** See
[Regenerate](#regenerate).

## Run

Every task bind-mounts its input data from `${{DATA_ROOT}}/Scenario-<N>/` on
the host, so `DATA_ROOT` (an **absolute** path -- Docker Compose resolves a
relative one against the task's `environment/` dir, not your shell's cwd, and
silently mounts empty directories instead of erroring) must be set on each
run command below. From the repo root:

```bash
uv run coding-agent-bench run \
    --agent claude-code \
    --dataset datasets/itbench-lite/harbor_tasks \
    --model-name my-model \
    --server-url http://my.server.url \
    --n-tasks 1 --envs DATA_ROOT={data_root} --dry-run
```

For example, to test locally against Ollama (bound to 127.0.0.1 by default,
unreachable from a container otherwise): uncomment `network_mode: host` in
`environment/docker-compose.yaml.tmpl` (undo before a real benchmarking run
against a proper model endpoint), then use `localhost`, not
`host.docker.internal` (that hostname doesn't resolve under host networking):

```bash
uv run coding-agent-bench run \
    --agent pi \
    --dataset datasets/itbench-lite/harbor_tasks \
    --model-name qwen3.5:2b \
    --server-url http://localhost:11434 \
    --model-max-len 32000 \
    --n-tasks 1 \
    --envs DATA_ROOT={data_root} --dry-run
```

Drop `--dry-run` to launch.

> [!note]
> Additional configuration options are available, use
> `uv run coding-agent-bench run --help` to see them.

Oracle (no model calls -- verifies task wiring):

```bash
DATA_ROOT={data_root} uv run harbor run -a oracle -p datasets/itbench-lite/harbor_tasks
```

## Regenerate

```bash
scripts/generate_custom_adapter_tasks.py itbench-lite --verify
```

(run from the repo root; there is no `generate_task.sh` in this checkout.)

{coverage_note} If the snapshot itself is incomplete, re-download it via the
HF CLI (`huggingface-cli download ibm-research/ITBench-Lite --repo-type
dataset --local-dir datasets/itbench-lite`) -- there is no `fetch_dataset.sh`
in this checkout either.
"""
