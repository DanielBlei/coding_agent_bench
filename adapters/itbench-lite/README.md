## ITBench-Lite → Harbor Adapter

Adapts the **SRE (Site Reliability Engineering) scenarios** of
[ibm-research/ITBench-Lite](https://huggingface.co/datasets/ibm-research/ITBench-Lite)
into Harbor tasks: offline Kubernetes root-cause analysis over exported
incident telemetry snapshots. FinOps and CISO scenarios are out of scope.

## Overview

- **Task type:** SRE fault localization / root-cause analysis (RCA). Each
  task gives the agent a static snapshot of a Kubernetes cluster captured
  during a simulated incident (alerts, metrics, k8s events/objects,
  OpenTelemetry logs/traces) and asks it to identify the single root-cause
  entity — the specific Kubernetes object whose failure or misconfiguration
  triggered the cascade, not a downstream symptom.
- **Domain/languages:** Kubernetes/observability telemetry; instructions in
  English.
- **Size/splits:** the upstream `sre` snapshot contains **35 scenarios**
  (`Scenario-N/` directories, non-contiguous numbering). This adapter
  generates one Harbor task per valid scenario; all 35 have been generated
  into `datasets/itbench-lite/harbor_tasks/` and the oracle scores
  `reward: 1.0` on every one of them.
- **Provenance:** [ITBench paper (arXiv:2502.05352)](https://arxiv.org/abs/2502.05352),
  [github.com/itbench-hub/ITBench](https://github.com/itbench-hub/ITBench)
  (scenario-generation framework behind this dataset), Apache-2.0.
- **Main adaptation modifications:**
  - Upstream ships **no problem statement for SRE scenarios** (only CISO
    scenarios carry a `problem.md`). `instruction.md` is therefore
    **synthesized** from the dataset card's task definition ("SRE: Fault
    Localization: identify the faulty entity or resource ... that caused the
    incident based on logs, traces, metrics, Kubernetes events and
    resources") plus an explicit answer contract.
  - Upstream scoring is **LLM-as-a-Judge**
    ([itbench-hub/ITBench-Evaluations](https://github.com/itbench-hub/ITBench-Evaluations));
    this adapter uses a **deterministic grader** instead (see
    [Scoring contract](#scoring-contract) and
    [Deviations from upstream](#deviations-from-upstream)).

## What is ITBench-Lite?

ITBench (IBM Research, ICML 2025) evaluates AI agents on real-world IT
automation tasks across SRE, FinOps, and CISO domains. ITBench-Lite is the
offline subset: SRE scenarios are exported snapshots of sandboxed live
Kubernetes environments in which a fault was injected and alerts fired. The
static format supports diagnostic analysis (identify the faulty entity) but
not live interaction or remediation. Upstream headline SRE metric is
`root_cause_entity` (precision/recall/F1 + pass@1 over the agent's predicted
entities), produced by an LLM judge comparing agent output against each
scenario's `ground_truth.yaml`.

## Adapter Features

- Enumerates `Scenario-N/` directories under the downloaded HF snapshot
  (`datasets/itbench-lite/snapshots/sre/*/`), validates each
  (`ground_truth.yaml` parses, has a `groups[]` entry with
  `root_cause: true`, all six input paths present), skips-and-logs invalid
  ones.
- **Answer-key isolation:** `ground_truth.yaml` lands only in `tests/`; the
  container bind-mounts the six input subpaths (`alerts/`, `metrics/`, the
  four root `*.tsv` files) read-only — never the scenario root.
- **Bind-mount inputs** via per-task `environment/docker-compose.yaml` from
  `${DATA_ROOT}` (multi-GB dataset; baking into images is impractical).
  Machine-local — see the generated runbook for `DATA_ROOT` usage.
- Per-scenario oracle `solution/solve.sh` writing the ground-truth answer.
- Deterministic multi-metric grader emitting `reward.json` (no network, no
  judge model).
- **Data-exploration tools baked into the agent image**, chosen for concrete
  format gotchas in this dataset (see [Scoring contract](#scoring-contract)
  / `instruction.md`'s "Available data" section for the gotchas themselves):
  `jq` (alerts JSON), `mlr`/miller (the four raw `*.tsv` files are
  RFC4180-quoted with embedded quotes/newlines in `Body`/attribute columns
  -- `awk`/`cut` misparse this, miller doesn't), `rg`/ripgrep (fast search
  over the largest files, up to a few hundred MB), and Python's `pandas`
  (also correctly parses the same quoting). None of these are used by the
  grader -- agent-only, baked at build time (network is available during
  `docker build`, same rationale as `curl`).

## Scoring contract

The agent writes `/workspace/answer.json`:

```json
{
  "root_cause": "<entity-name>",
  "kind": "<kind>",
  "reasoning": "<what/why/how, with telemetry evidence>",
  "propagation_chain": ["<root-entity>", "...", "<symptom-entity>"]
}
```

(contract stated in `instruction.md`). `tests/grade.py` compares it against
the root-cause group of `tests/ground_truth.yaml`:

| Metric | Definition |
|---|---|
| `reward` (headline) | 1.0 iff `kind_match == 1.0` AND `name_match == 1.0` AND the answer was properly written to `answer.json`; halved to 0.5 if that same correct answer was only recovered from the transcript (`answer_recovered_from_transcript == 1.0`) — partial credit for reasoning right but not completing the file-write requirement; 0.0 if `kind_match`/`name_match` fail regardless of source |
| `kind_match` | answer `kind` equals the root group's `kind`, case-insensitive |
| `name_match` | answer `root_cause` matches (`re.search`) at least one of the root group's patterns — its `filter[]` regexes, or (fallback, e.g. a ConfigMap root cause with no `filter`) an anchored exact match on its `name` |
| `answer_format_valid` | 1.0 iff `answer.json` exists, parses, and has both string keys (mechanical file-write signal) |
| `answer_recovered_from_transcript` | 1.0 iff a missing/malformed answer file was compensated by best-effort recovery of a JSON blob from the agent's transcript under `/logs/agent/`; always 0.0 when `answer_format_valid` is 1.0. Diagnostic, not a free pass — see `reward`'s recovery penalty |
| `chain_proximity` | informational only, never affects reward: 1.0 iff the answer's `root_cause` matches the patterns of any group referenced in `propagations[].source/.target` |
| `reasoning_present` | informational only: 1.0 iff `reasoning` is a string of >= 20 characters after stripping — did the agent articulate what/why/how, not just where |
| `propagation_chain_coverage` | informational only, in [0, 1]: fraction of the distinct groups referenced by `propagations[].source/.target` whose patterns match at least one entry of the agent's declared `propagation_chain` |

Only `kind_match`/`name_match` (hence `reward`) and `answer_format_valid`
gate pass/fail. `reasoning`/`propagation_chain` are captured to see *whether
the agent understood the incident*, not just whether it named the right
object — `grade.py` also echoes the agent's `reasoning` and
`propagation_chain` verbatim to verifier stdout (`test-stdout.txt` in job
artifacts) for qualitative review, e.g. distinguishing "found entity X" from
"understood the whole cascade but named the wrong root".

### Metric parity with upstream

This adapter's informational metrics are deterministic proxies for specific
upstream LLM-judge metrics
([`ITBench-Evaluations`](https://github.com/itbench-hub/ITBench-Evaluations/blob/main/README.md#metrics-covered)):

| This adapter (`grade.py`) | Nearest upstream metric | Note |
|---|---|---|
| `kind_match` + `name_match` (→ `reward`) | `root_cause_entity` (pass@1) | Closest real analogue; `reward` is the headline comparison point |
| `reasoning_present` | `root_cause_reasoning` | Presence/length proxy vs. judged correctness |
| `chain_proximity` | `root_cause_proximity` / `root_cause_proximity_with_fp` | Binary match-in-graph vs. judged hop-distance |
| `propagation_chain_coverage` | `propagation_chain` | Regex coverage fraction vs. judged full-chain scoring |
| *(none)* | `fault_localization_component_identification` | No analogue — upstream scores "first symptom identified," not modeled here |
| *(none)* | `root_cause_reasoning_partial` | No analogue — partial credit for correctly diagnosing a downstream symptom when the root cause is missed |

## Deviations from upstream

- **Deterministic grader instead of LLM-as-a-Judge.** Upstream's
  ITBench-Evaluations uses a judge model to normalize entities and score
  `root_cause_entity` (P/R/F1 + pass@1), `root_cause_reasoning`,
  `propagation_chain`, proximity, etc. Harbor grading must be reproducible
  and network-free, so this adapter scores the headline `reward` on a single
  declared answer deterministically — closest analogue to upstream
  **pass@1** on `root_cause_entity`. `reasoning_present` and
  `propagation_chain_coverage` are deterministic, informational proxies for
  upstream's judged `root_cause_reasoning`/`propagation_chain` (presence and
  regex coverage rather than judged correctness of the prose) — they never
  gate `reward`, but are echoed to verifier stdout so a human/reviewer can
  see *whether the agent understood the incident*, not just whether it named
  the right object.
- **Root-cause groups without a `filter[]`.** A few scenarios (e.g.
  Scenario-2's `flagd-config` ConfigMap root cause) give the root-cause
  group only a `name`, no `filter` regexes. `grade.py`'s `group_patterns()`
  falls back to an anchored exact match on `name` in that case; the adapter
  skips-and-logs any scenario whose root-cause group has neither.
- **`aliases` are deliberately never consulted.** Upstream's judge rubric
  counts any alias of a root-cause entity as correct. In this dataset the
  alias groups can mix the root cause with downstream propagated/symptom
  entities (e.g. Scenario-1's single alias group contains both the
  `load-generator` root cause and the `frontend-proxy` symptom entities), so
  honoring aliases would score symptom answers as correct. Expect Harbor
  scores to be **systematically stricter** than upstream `root_cause_entity`
  on answers that name an alias rather than the root-cause entity itself.
- **Synthesized instructions** (see Overview) — upstream provides none for
  SRE.
- Entity name format: upstream agent outputs use `namespace/Kind/name`;
  here the answer is a flat `{root_cause, kind}` object where `root_cause`
  is the name as it appears in the telemetry.
- **`recommended_actions` is intentionally unused.** The ground truth's
  remediation playbook backs the live-environment "remediate the incident"
  task, which the offline snapshots can't support (dataset card: no active
  remediation). Upstream's Lite evaluation scores diagnosis only (no
  remediation metric), and free-text actions aren't deterministically
  gradable, so the grader never reads this section.
- **No pre-built application-topology fixture.** IBM's own reference agent
  ([itbench-hub/ITBench-CISO-SRE-FinOps-Agent](https://github.com/itbench-hub/ITBench-CISO-SRE-FinOps-Agent))
  ships a hand-authored `architecture.json` service-dependency graph, mounted
  into every scenario's workspace and consumed by dedicated
  `build_topology`/`topology_analysis` tools. That fixture isn't part of the
  `ibm-research/ITBench-Lite` HF dataset and this adapter doesn't provide an
  equivalent — the agent must reconstruct service topology from raw
  `k8s_objects_raw.tsv`/trace data every scenario, from scratch. This is a
  deliberate scope choice, not an oversight: this adapter also tests
  topology reconstruction from raw telemetry, not just fault diagnosis given
  a known topology. It does mean scores here are not apples-to-apples with
  the official ITBench leaderboard (which was produced against the
  reference agent's topology-assisted setup).
- **Answer schema is narrower than upstream's, not just flatter.** The
  reference agent's mandatory output is a graph supporting **multiple
  independent root causes** (`entities[]` with `contributing_factor: true`,
  gated by an "irreducibility test") and requires **every observed alert to
  be accounted for** (`alerts_explained[]`). This adapter's single-answer
  contract has dropped both the multi-cause case and the
  alert-coverage requirement entirely, not just simplified a list down to
  one item — a scenario with more than one independent fault is scored here
  purely on whether the *primary* root cause was named, with no signal about
  whether other independent causes or unexplained alerts were missed.
- **No tool-assist layer, by design — and it changes what's being measured.**
  The reference agent invests in a 10-tool MCP server
  (`offline_incident_analysis`; pre-aggregated alert/event/metric queries,
  anomaly detection, topology queries, config-drift detection). This adapter
  instead bakes in generic CLI tools (`jq`, `mlr`, `rg`, `pandas`) and
  expects the agent to write its own ad hoc parsing/joining logic every run
  (see "Data-exploration tools" above). This keeps the task self-contained
  in a single Docker image with no MCP server to version or maintain, but it
  means the two setups aren't measuring quite the same skill: reference-agent
  scores partly reflect "used the right pre-built tool call," while this
  adapter's scores partly reflect "wrote correct pandas/jq/mlr code under
  time pressure" — a real skill, but a different one, adding variance
  unrelated to diagnostic reasoning quality.

## Generated Task Structure

```
datasets/itbench-lite/harbor_tasks/
├── README.md                     # generated runbook (run/regenerate)
└── scenario-<N>/
    ├── task.toml                 # Task configuration
    ├── instruction.md            # Synthesized prompt + answer contract
    ├── environment/
    │   ├── Dockerfile
    │   └── docker-compose.yaml   # bind-mounts the six input subpaths from ${DATA_ROOT}
    ├── solution/
    │   └── solve.sh              # writes the ground-truth answer.json
    └── tests/
        ├── test.sh               # runs grade.py
        ├── grade.py              # deterministic grader (see scoring contract)
        └── ground_truth.yaml     # answer key, grader-only
```

The adapter code lives at `adapters/itbench-lite/`:

```
adapters/itbench-lite/
├── README.md
├── adapter_metadata.json
├── parity_experiment.json
├── pyproject.toml
└── src/itbench_lite/
    ├── __init__.py
    ├── adapter.py
    ├── main.py
    └── task-template/
        ├── task.toml
        ├── instruction.md
        ├── environment/
        │   ├── Dockerfile
        │   └── docker-compose.yaml.tmpl
        ├── solution/
        │   └── solve.sh
        └── tests/
            ├── grade.py
            └── test.sh
```

## Run Evaluation / Harness

### Recommended: this repo's `coding-agent-bench` CLI

For real agent+model runs (not just the oracle), prefer the repo's own
wrapper over raw `harbor run` — it builds the equivalent command for you and
wires up agent-specific config (e.g. the `models.json` mount for `pi`). This
adapter also bind-mounts scenario data at run time, so pass `DATA_ROOT` via
`--envs` — see `datasets/itbench-lite/harbor_tasks/README.md` for the exact
value.

```bash
uv run coding-agent-bench run \
  --agent <agent> \
  --dataset datasets/itbench-lite/harbor_tasks \
  --model-name <model> \
  --server-url <model-server-url> \
  --model-max-len <context-length> \
  --n-tasks 1 --envs DATA_ROOT="<snapshot dir -- see harbor_tasks/README.md>" \
  --dry-run   # drop --dry-run to actually launch
```

- `--agent` — the harness to run: one of `oracle`, `claude-code`, `codex`,
  `openclaw`, `opencode`, `pi`.
- `--dataset` — pass this **local path**, not the HF dataset name. This
  adapter isn't registered in Harbor's dataset registry;
  `coding-agent-bench` builds `-p <path>` instead of `-d <name>` whenever
  the given path exists on disk, resolved relative to the current directory
  — always run from the repo root.
- Always run with `--dry-run` first to print the resulting `harbor run ...`
  command before launching for real.

### Using Job Configurations

```bash
# From the repository root
uv run harbor run -c adapters/itbench-lite/run_itbench-lite.yaml
# Or with a locally prepared dataset path:
uv run harbor run -p datasets/itbench-lite/harbor_tasks -a <agent_name>
```

Results are saved in the `jobs/` directory by default (configurable via
`jobs_dir` in the YAML config). Note every invocation needs `DATA_ROOT` set
(see the generated runbook).

## Usage: Create Task Directories

```bash
cd adapters/itbench-lite
uv run itbench-lite --output-dir ../../datasets/itbench-lite/harbor_tasks --overwrite
```

Available flags:
- `--output-dir` — Directory to write generated tasks (defaults to
  `datasets/itbench-lite` at the repo root; the repo convention is
  `datasets/itbench-lite/harbor_tasks`)
- `--limit` — Generate only the first N tasks
- `--overwrite` — Overwrite existing tasks
- `--task-ids` — Only generate specific task IDs (e.g. `scenario-1`)

## Comparison with Original Benchmark (Parity)

See `parity_experiment.json` (pending). Headline comparison metric is
upstream `root_cause_entity` **pass@1** vs. Harbor `reward`; note the
[deviations above](#deviations-from-upstream) — in particular the strict
alias policy — make exact numeric parity with the upstream LLM-judge
pipeline neither expected nor required for a faithful deterministic
adaptation.

## Notes & Caveats

- **Inputs are bind-mounted, not baked**: tasks are machine-local and
  require `DATA_ROOT` (absolute path) at run time — see the generated
  runbook. Not publishable to the Harbor dataset registry as-is.
- The full sre snapshot (~35 scenarios / multi-GB) is downloaded on disk and
  all 35 have been generated as Harbor tasks.
- Grader runs in the task image (`python:3.13-slim` + pyyaml); no network
  needed for grading. The same image also carries agent-only exploration
  tools (`jq`, `mlr`, `rg`, `pandas`) -- image is ~450MB, mostly
  pandas/numpy; build time ~20s, well under `build_timeout_sec = 600.0`.
- `chain_proximity`, `reasoning_present`, and `propagation_chain_coverage`
  are informational only and never affect `reward`.

## Installation / Prerequisites

- Docker installed and running; Harbor working (see main repository README).
- Adapter dependencies:
  ```bash
  cd adapters/itbench-lite
  uv sync
  ```
- Dataset download (sample or full): this checkout has no
  `.claude/skills/new-adapter/scripts/fetch_dataset.sh` (referenced by older
  docs/skills, not present here) — use the standard HF CLI instead:
  ```bash
  # Sample (single scenario):
  huggingface-cli download ibm-research/ITBench-Lite --repo-type dataset \
      --include "snapshots/sre/v0.2-*/Scenario-1/*" \
      --local-dir datasets/itbench-lite
  # Full snapshot:
  huggingface-cli download ibm-research/ITBench-Lite --repo-type dataset \
      --local-dir datasets/itbench-lite
  ```

## Troubleshooting

- **Agent sees empty input dirs**: `DATA_ROOT` was unset or relative —
  Compose resolves relative paths against the task's `environment/` dir and
  silently mounts auto-created empty dirs. Use the absolute value from the
  generated runbook.
- **Oracle fails**: ensure the task was regenerated after adapter changes
  (`scripts/generate_custom_adapter_tasks.py itbench-lite --verify`, run
  from the repo root; there is no `generate_task.sh` in this checkout).
- **`expected exactly one snapshot-version dir`**: the adapter expects a
  single `snapshots/sre/<version>/` directory; prune stale ones.

## Citation

```bibtex
@inproceedings{jha2025itbench,
  title     = {{ITB}ench: Evaluating AI Agents across Diverse Real-World IT Automation Tasks},
  author    = {Jha, Saurabh and Arora, Rohan and Watanabe, Yuji and others},
  booktitle = {Forty-second International Conference on Machine Learning},
  year      = {2025},
  url       = {https://openreview.net/forum?id=jP59rz1bZk}
}
```
