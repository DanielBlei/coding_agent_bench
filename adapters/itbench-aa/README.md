## ITBench-AA → Harbor Adapter

Converts the public SRE (Site Reliability Engineering) scenarios of
[ArtificialAnalysis/ITBench-AA](https://huggingface.co/datasets/ArtificialAnalysis/ITBench-AA)
(CC-BY-4.0, built in partnership with IBM) into Harbor tasks: offline
Kubernetes root-cause analysis over exported incident telemetry snapshots.
Same data family/ground-truth schema as
[`adapters/itbench-lite`](../itbench-lite) (which adapts
`ibm-research/ITBench-Lite` instead) — independent adapter, no shared code.

Each scenario is a static snapshot of a Kubernetes cluster mid-incident —
alerts, metrics, k8s events/objects, OpenTelemetry logs/traces — captured
after a fault was injected. The agent's job: identify the root-cause
entity (or entities), not a downstream symptom. 40 of the upstream split's
59 scenarios are public (the other 19 are held out); this adapter generates
one Harbor task per valid `Scenario-N/` present under the downloaded
snapshot.

## Run evaluation

### Recommended: `coding-agent-bench` CLI

Only this repo's custom Podman environment auto-applies a task's
`environment/mounts.json` today (its own `mounts.json` bakes in absolute
host paths at generation time — no `DATA_ROOT` needed), so pass
`--environment podman`; add `--host-network` to reach a host-local model
server (e.g. Ollama):

```bash
uv run coding-agent-bench run \
  --agent <agent> \
  --dataset datasets/itbench-aa/harbor_tasks \
  --model-name <model> \
  --server-url <model-server-url> \
  --model-max-len <context-length> \
  --environment podman --host-network \
  --n-tasks 1 \
  --dry-run   # drop --dry-run to actually launch
```

- `--agent` — one of `oracle`, `claude-code`, `codex`, `openclaw`,
  `opencode`, `pi`.
- `--dataset` — this **local path**, not an HF dataset name (this adapter
  isn't in Harbor's dataset registry). Run from the repo root.

### Raw `harbor run`

```bash
uv run harbor run -c adapters/itbench-aa/run_itbench-aa.yaml
# or:
uv run harbor run -p datasets/itbench-aa/harbor_tasks -a <agent_name>
# On Podman instead of Docker, add:
#   --env coding_agent_bench.helpers.podman:PodmanEnvironment
```

### Generate / regenerate tasks

```bash
scripts/generate_custom_adapter_tasks.py itbench-aa --verify
```

Runs from the repo root; generates every valid scenario found under the
downloaded snapshot into `datasets/itbench-aa/harbor_tasks/`, then (with
`--verify`) replays the oracle and requires `reward: 1.0` on each. Flags:
`--task-ids <id>` (one task only), `--overwrite`, `--podman` (force Podman
for the verify run), `--host-network`. Equivalent direct invocation:

```bash
cd adapters/itbench-aa
uv run itbench-aa --output-dir ../../datasets/itbench-aa/harbor_tasks \
  --overwrite [--limit N] [--task-ids scenario-1 ...] [--data-root <path>]
```

`--data-root` overrides dataset-root auto-detection (HF Hub cache, falling
back to `datasets/itbench-aa/`) — use it if your download landed somewhere
else or in a different layout.

## What's delivered

- **Task**: SRE fault localization. The agent reads read-only telemetry
  under `/workspace/` and writes its diagnosis to `/workspace/answer.json`
  (full contract in the generated `instruction.md`):

  ```json
  {
    "root_causes": [
      {"name": "<entity-name>", "kind": "<kind>", "namespace": "<namespace>"}
    ],
    "reasoning": "<what/why/how, with telemetry evidence>",
    "propagation_chain": ["<root-entity>", "...", "<symptom-entity>"],
    "recommended_actions": ["<action>"]
  }
  ```

- **Grading**: `tests/grade.py` is deterministic (no judge model, no
  network) and compares the answer against `tests/ground_truth.yaml`'s
  `groups[]` (every group with `root_cause: true`). Headline `reward` is
  recall-gated precision: `0.0` unless *every* ground-truth root-cause
  group is matched by some submitted entity, in which case `reward =
  matched_count / submitted_entity_count` — padding the list with
  unmatched entities lowers your score even once recall is satisfied.

  | Metric | Definition |
  |---|---|
  | `reward` (headline) | `0.0` if `root_causes` is empty, the answer was recovered from the transcript, or any ground-truth root-cause group is unmatched; else `matched_count / submitted_entity_count` |
  | `name_match` | best match across submitted entities: anchored full-string match against a ground-truth group's `filter[]` patterns, or an anchored exact match on `name` as fallback — alias-aware |
  | `kind_match` | best match across submitted entities: `kind` equals the matched group's `kind`, case-insensitive |
  | `namespace_applicable` / `namespace_match` | whether the matched group has a namespace, and whether the submission's matches; **a mismatch blocks claiming that group**, feeding the recall gate on `reward` |
  | `answer_format_valid` | `answer.json` exists, parses, `schema_version` absent or `"1.0"`, `root_causes` non-empty |
  | `answer_recovered_from_transcript` | a missing/malformed `answer.json` was recovered from the agent's transcript instead; always forces `reward = 0.0` |
  | `chain_applicable` / `chain_head_correct` / `propagation_edge_coverage` / `chain_resolution_rate` | diagnostics on the submitted `propagation_chain` vs. `ground_truth.yaml`'s `propagations[]`; never gate `reward` |
  | `reasoning_present` | any token in `reasoning` resolves to a ground-truth group; relevance check, not length/quality |
  | `submitted_entity_count` / `turn_count` / `recommended_actions` | diagnostic only, echoed to verifier stdout, never scored |

  Only `reward` and `answer_format_valid` gate pass/fail.

- **Answer-key isolation**: `ground_truth.yaml` lands only in `tests/`; the
  container only ever sees the six input subpaths (`alerts/`, `metrics/`,
  the four root `*.tsv` files), read-only, never the scenario root.
- **Data-exploration tools baked into the agent image**: `jq` (alerts
  JSON), `mlr`/miller and `pandas` (the four raw `*.tsv` files are
  RFC4180-quoted with embedded quotes/newlines — `awk`/`cut` misparse
  this), `rg`/ripgrep. Agent-only, not used by the grader.

Generated task layout:

```
datasets/itbench-aa/harbor_tasks/
├── README.md                     # generated runbook (run/regenerate)
└── scenario-<N>/
    ├── task.toml
    ├── instruction.md             # what the agent reads
    ├── environment/
    │   ├── Dockerfile
    │   ├── mounts.json            # bind-mount manifest, absolute host paths
    │   └── host-network-overlay.yaml.example
    ├── solution/solve.sh          # oracle: writes the ground-truth answer
    └── tests/
        ├── test.sh                # runs grade.py
        ├── grade.py               # deterministic grader
        └── ground_truth.yaml      # answer key, grader-only
```

## Installation / Prerequisites

- Docker or Podman installed and running; Harbor working (see main
  repository README).
- Adapter dependencies: `cd adapters/itbench-aa && uv sync`
- Dataset download (layout is **flat**, `sre/Scenario-N/...`):
  ```bash
  hf download ArtificialAnalysis/ITBench-AA --repo-type dataset \
      --local-dir datasets/itbench-aa
  ```
  Pass `--data-root <path>` when generating tasks if you downloaded
  somewhere else.

## Troubleshooting

- **Agent sees empty input dirs**: expected on Docker today — only the
  custom Podman environment auto-applies a task's `environment/mounts.json`;
  run with `--environment podman`, or build `--mounts` from that file by
  hand.
- **Oracle fails**: regenerate after any adapter change
  (`scripts/generate_custom_adapter_tasks.py itbench-aa --verify`).
- **`could not find scenario data under ...`**: dataset root resolution
  tried the flat (`<root>/sre/`) and wrapped/versioned
  (`<root>/snapshots/sre/<one-dir>/`) layouts and found no `Scenario-*`
  dirs in either — the error lists every path checked. Fix the download
  location or pass `--data-root <path>`.

## Citation

Direct data source:
[ArtificialAnalysis/ITBench-AA](https://huggingface.co/datasets/ArtificialAnalysis/ITBench-AA)
(CC-BY-4.0) — cite it alongside the underlying ITBench methodology/paper:

```bibtex
@inproceedings{jha2025itbench,
  title     = {{ITB}ench: Evaluating AI Agents across Diverse Real-World IT Automation Tasks},
  author    = {Jha, Saurabh and Arora, Rohan and Watanabe, Yuji and others},
  booktitle = {Forty-second International Conference on Machine Learning},
  year      = {2025},
  url       = {https://openreview.net/forum?id=jP59rz1bZk}
}
```
