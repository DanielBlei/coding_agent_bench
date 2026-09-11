"""
Generates datasets/itbench-aa/harbor_tasks/README.md -- the runbook shown
to a human running the generated tasks (not the agent; see task-template/
instruction.md for that). Rewritten on every `run()`, so it always reflects
the current snapshot location and generated-task coverage.
"""

from __future__ import annotations

from pathlib import Path

from ._common import REPO_ROOT


def write_runbook(
    output_dir: Path, snapshot_root: Path, total_valid_scenarios: int, existing_task_count: int
) -> None:
    readme_path = output_dir / "README.md"
    readme_path.write_text(
        _runbook_text(snapshot_root, total_valid_scenarios, existing_task_count)
    )


def _runbook_text(
    snapshot_root: Path, total_valid_scenarios: int, existing_task_count: int
) -> str:
    try:
        snapshot_root_rel: Path | str = snapshot_root.relative_to(REPO_ROOT)
    except ValueError:
        snapshot_root_rel = snapshot_root
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
    return rf"""# itbench-aa generated tasks

**Source:** [ArtificialAnalysis/ITBench-AA](https://huggingface.co/datasets/ArtificialAnalysis/ITBench-AA)
(public split, built in partnership with IBM; CC-BY-4.0)

**This directory is generated. Do not hand-edit it.** See
[Regenerate](#regenerate).

## Run

Each task ships `environment/mounts.json` with its scenario's input data
already resolved to absolute host paths at generation time -- no
`DATA_ROOT`, no manual `--mounts`. Podman applies this automatically (via
this repo's custom `coding_agent_bench.helpers.podman:PodmanEnvironment`);
Docker's built-in environment does not read this file, so a real (non-oracle)
Docker run needs its mounts built by hand -- Podman is the zero-config path
today.

From the repo root, on Podman against a local Ollama server:

```bash
uv run coding-agent-bench run \
    --agent pi \
    --dataset datasets/itbench-aa/harbor_tasks \
    --model-name qwen3.8:27b \
    --server-url http://localhost:11434 \
    --environment podman --host-network \
    --n-tasks 1 --dry-run
```

`--host-network` lets the container reach a host-local model server (e.g.
Ollama, bound to 127.0.0.1 by default) -- use `localhost`, not
`host.docker.internal`. Drop `--dry-run` to launch.

> [!note]
> Additional configuration options are available, use
> `uv run coding-agent-bench run --help` to see them.

Oracle (no model calls, no input mounts needed -- verifies task wiring):

```bash
uv run harbor run -a oracle -p datasets/itbench-aa/harbor_tasks
# On Podman instead of Docker:
#   --env coding_agent_bench.helpers.podman:PodmanEnvironment
```

## Regenerate

```bash
scripts/generate_custom_adapter_tasks.py itbench-aa --verify
```

(run from the repo root; there is no `generate_task.sh` in this checkout.)
`--verify` re-runs the Harbor oracle and requires reward = 1; it
auto-detects Docker vs. Podman (pass `--podman` to force Podman, or
`--host-network` if the oracle itself needs to reach a model server).

{coverage_note} If the snapshot itself is incomplete, re-download it via the
HF CLI (`hf download ArtificialAnalysis/ITBench-AA --repo-type dataset
--local-dir datasets/itbench-aa`) -- there is no `fetch_dataset.sh` in this
checkout either. If your download landed somewhere else or in a different
layout, pass `--data-root <path>` to `itbench-aa` instead of relying on
auto-detection.
"""
