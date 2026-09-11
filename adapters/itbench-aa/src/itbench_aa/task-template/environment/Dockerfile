FROM docker.io/library/python:3.13-slim

WORKDIR /workspace

# Minimal base image: no scenario data is baked in here. Scenario inputs are
# bind-mounted at runtime via `harbor run --mounts` (see
# environment/mounts.json and datasets/itbench-aa/harbor_tasks/README.md) --
# baking would mean ~38GB across the full ~35-scenario dataset.
#
# pyyaml is needed by tests/test.sh to parse the answer key at grading
# time. The verifier runs inside this same image, so it must be installed
# here rather than only in the adapter's own pyproject.toml.
#
# pandas is for the agent: the four raw *.tsv files (k8s_events, k8s_objects,
# otel_logs, otel_traces) are RFC4180-quoted with embedded quotes/newlines
# (see instruction.md) -- pandas.read_csv(sep="\t") parses this correctly,
# unlike naive line-splitting. Not needed by the grader itself.
RUN pip install --no-cache-dir pyyaml pandas

# curl is baked in at build time (network available during `docker build`)
# rather than left for the agent harness to apt-get at container start --
# task.toml sets network_mode = "public" for the agent phase (see task.toml),
# but Harbor's per-agent installers (e.g. pi) probe `command -v curl` before
# attempting to install it, so pre-baking it here makes that probe succeed
# and skips the network-dependent path entirely.
#
# Agent-facing data-exploration tools -- all here because of concrete format
# gotchas in this dataset (see instruction.md's "Available data" section):
#   jq       - alerts/*.json are plain JSON; trivial to filter/query.
#   mlr      - the *.tsv files are RFC4180-quoted (embedded quotes/newlines
#              in the Body/attribute columns); miller respects that quoting
#              correctly (unlike awk/cut/grep splitting on tab), handles
#              large files fast, and can convert TSV rows to JSON to pipe
#              into jq (`mlr --itsv --ojson cat file.tsv | jq ...`).
#   ripgrep  - fast full-text search across the largest files (otel_logs_raw
#              .tsv / otel_traces_raw.tsv can be hundreds of MB).
# None of these are used by the grader -- agent-only.
RUN apt-get update && apt-get install -y --no-install-recommends \
        jq curl miller ripgrep git \
    && rm -rf /var/lib/apt/lists/*
