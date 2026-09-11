# ITBench-AA Scoring Deviations and Grading Notes

**Last updated**: 2026-09-10  
**Dataset revision**: `0ad7ce237f82` — patches are keyed on exact filter strings;
if upstream edits the source YAML those patches silently no-op. Re-verify after
any dataset update.

**Open calibration runs**: Stirrup calibration: not yet done. Upstream issue
filing: not yet done (see Data quality section).

This file documents how this adapter's scoring contract relates to the two
upstream references, **plus one architectural deviation** (data delivery; see
next section). Scoring references: **Artificial Analysis (AA)** at
artificialanalysis.ai/evaluations/itbench-aa (live; 59 tasks: 40 from IBM's
public release plus 19 private tasks shared by the ITBench team), and **IBM
upstream** at itbench-hub/ITBench (the public dataset this adapter is built on).

**Coverage caveat**: this adapter runs 40 of AA's 59 scenarios (the public IBM
release). The 19 private AA tasks are not included. A run on this adapter
cannot be compared directly to AA's headline number; it covers a strict subset.

---

## Architectural deviation: runtime bind mounts vs. baked-in data

**Status: not yet addressed. Tracked as a future task — align with the other
adapters / the standard workflow (SWE-bench-style).**

Unlike SWE-bench and other standard Harbor adapters, this adapter does **not**
ship self-contained task environments. Standard adapters bake everything the
task needs into the image — either a prebuilt `docker_image` per task instance,
or `COPY`/`git clone` in the task Dockerfile — so the container has no
dependency on the host it runs on and behaves identically on Docker, Podman, a
remote machine, or OpenShift.

This adapter instead **bind-mounts scenario data from the host at runtime**. The
Dockerfile deliberately bakes in no data (baking would be ~38GB across the full
~35-scenario dataset), so each task writes a per-task `environment/mounts.json`
with absolute host `source` paths (into the HF Hub cache) that are mounted into
`/workspace` when the container starts.

Consequences of this deviation (all of which the SWE-bench-style approach avoids):

- **No per-task mount hook in Harbor.** Harbor's task-level `EnvironmentConfig`
  (`task.toml [environment]`) has no `mounts` field, and the `--mounts` /
  `--mounts-json` CLI flag is **job-wide** (one array for every task). Since each
  scenario mounts a different set of files (distinct HF blob hashes, per-minute
  alert filenames), a job-wide array cannot serve a multi-scenario run. The only
  per-task injection point is a custom environment class constructed with a
  per-task `environment_dir`.
- **Only Podman is wired.** `coding_agent_bench.helpers.podman:PodmanEnvironment`
  reads the per-task `mounts.json`; Harbor's stock Docker environment does not,
  so a `--environment docker` run gets an **empty `/workspace`** and the agent
  cannot see any telemetry. Running this dataset on Docker today requires either
  adding a parallel custom Docker environment class or a per-task
  `environment/docker-compose.yaml` (the itbench-lite approach) — neither of
  which is in place.
- **Not portable to remote / OpenShift.** The `source` paths point at the local
  host's `~/.cache/huggingface/...`; on any other machine (remote runner,
  OpenShift node) those paths don't exist and the mount fails regardless of the
  container engine.

**Intended resolution (future task):** move to the standard, self-contained
model — bake scenario data into per-task images (or publish prebuilt per-scenario
`docker_image`s / a data layer), dropping runtime bind mounts entirely. That
removes `mounts.json`, the custom Podman environment dependency for data
delivery, and the Docker/Podman/remote divergence, and lets itbench-aa run
through the same workflow as every other adapter.

As part of that migration, **`scripts/generate_custom_adapter_tasks.py` will be
dropped** — it is a repo-local helper that exists only because this adapter is
generated and mounted outside the standard flow (it hand-builds the `--mounts`
flag and picks the container engine for oracle verification). Under the
SWE-bench-style adapter, generation/verification is the adapter's own
responsibility, so that script's actions would be migrated into the adapter
itself and the standalone script removed.

---

## Scoring contract

### 1. Name matching

**vs. IBM upstream**: IBM's stated methodology involves ground truth filter
regexes. Whether IBM applies additional LLM-based assessment, normalises string
form, or uses the filter regexes as a hard gate has not been independently
confirmed. The scoring rule used here — `re.fullmatch` against the filter
patterns — is one plausible interpretation of the ground truth schema, not a
verified implementation of IBM's scorer.

**vs. AA**: AA publishes the headline metric but not the entity-matching rule.
The AA scorer is not public. Whether AA normalises casing, resolves through
aliases, or applies any string-form tolerance is unknown. The direction of any
deviation between our fullmatch rule and AA's rule is unknown, not just the
magnitude. Do not assume our score will be higher or lower than AA's for a given
model until a calibration run establishes the relationship.

### 2. Recall-gated precision headline

`reward = |matched GT root-cause groups| / |submitted root_causes|`, gated on
full recall: any unmatched GT group forces reward to 0.0 regardless of other
correct identifications.

**vs. AA**: AA publishes this as its headline metric. This is parity, not a
deviation.

**vs. IBM upstream**: IBM's single-entity answer format has no equivalent
concept; the precision denominator and recall gate are additions relative to IBM.

### 3. Transcript-recovered answers excluded from headline

If an agent states the correct answer in chat but never writes
`/workspace/answer.json`, this adapter scores `reward=0.0`.
`answer_recovered_from_transcript` is reported as a separate per-scenario
metric so the failure mode is visible across a run.

**vs. AA**: AA scores the written output file. Our exclusion is close to parity.

**vs. IBM upstream**: Unknown — not confirmed whether IBM's evaluation reads the
full transcript or only the written file.

### 4. Scoring rule disclosed in agent prompt

`instruction.md` includes the full answer schema and enforcement-class table.
AA's task prompt is not public, so whether it discloses the scoring rule is
unknown. Our disclosure is a deliberate choice and a candidate confound when
comparing scores to AA.

---

## Data quality issues found during adapter development

The three findings below should be filed as a single issue at
itbench-hub/ITBench. **Status: not yet filed.**

These were found by running `derive_matching_literal(strict=True)` on every
root-cause group and cross-referencing synthesised literals against the cluster
topology in the ground truth.

### Scenario-29: filter `.*adservice` matches no cluster resource

The root-cause filter was `.*adservice`. The OTel Demo cluster names that service
`ad` (pods: `ad-*`), not `adservice`. Any agent correctly identifying the `ad`
Deployment would score `name_match=0` against the raw filter.

**Resolution**: the `ground_truth.yaml` is copied verbatim into the generated
task (`adapter.py._render_task`); filters are **not** rewritten at generation
time. The oracle stays gradeable because `oracle._oracle_answer()` derives its
answer with `matching.derive_matching_literal(strict=True)`, which synthesises a
literal (`adservice`) that fullmatches `.*adservice` — so the oracle scores
1.0 (verified by `tests/test_oracle_end_to_end.py`). **Open, agent-facing:** an
agent naming the real cluster resource `ad` still fails, because
`grade.py` scores with `name_matches_strict` (anchored fullmatch, no
service-name aliasing). This service-name gap is separate from the
controller↔Pod widening below and remains open.

> **Historical note:** earlier revisions of this file described a
> `adapter.py._KNOWN_FILTER_PATCHES` table and a `tests/patches_applied.json`
> writer that rewrote filters at generation time. **That mechanism does not
> exist in the current code** — `_render_task` copies `ground_truth.yaml`
> unchanged. The references below have been corrected to match reality.

### Scenario-20: group id does not fullmatch its own filter

The root-cause group id is `productcatalog-deployment-1` (no hyphen) while the
filter is `product-catalog-.*` (with hyphen). `_synthesize_literal` derives
`product-catalog-` (trailing hyphen) rather than the clean name `product-catalog`.
The filter is correct for the cluster; the id is a naming inconsistency in the
ground truth.

**Resolution**: no filter rewrite is needed. The oracle's
`derive_matching_literal(strict=True)` synthesises `product-catalog` from the
filter `product-catalog-.*` (its `name_matches_root_identity`-style suffix
relaxation strips the trailing `-.*`), which fullmatches — so the oracle scores
1.0 regardless of the id/filter naming inconsistency.

### Scenarios 23 and 105: filter too narrow for bare Deployment name

The root-cause group in each is kind=Deployment. The original filters
(`checkout-.*`, `product-catalog-.*`) match pod-suffixed names (e.g.
`checkout-7f9c-x2k`) but not the bare Deployment name (`checkout`). An agent
correctly identifying the root cause as `checkout` (kind=Deployment) fails
`name_match` against the raw filter.

Separately, the `aliases` block links Service and Pod representations but does
not always include the Deployment. Historically an agent submitting a
workload-equivalent kind (e.g. `Deployment` against a Pod-modelled root, or
`Pod` against a Deployment-modelled root) with the correct name could not reach
the root cause.

**Resolution for name matching**: no filter rewrite is needed. `grade.py` scores
the bare workload name via `name_matches_root_identity`, which fullmatches after
stripping a trailing `-.*`/`-.+` from the root filter (so `checkout` matches
`checkout-.*`). **Resolution for the kind gap**: the controller↔Pod↔Service
*workload equivalence* below lets `kind=Deployment name=checkout` (or `kind=Pod`)
claim the root, gated on the submitted name matching the **root group's own**
suffix-relaxed pattern — so it never lets a different workload through. See
"Controller↔Pod workload equivalence".

---

## Grading nuances and metric interpretation

### Conditioned metric aggregation

Harbor's aggregator averages every emitted metric over all N scenarios. For
metrics with an applicability flag, the raw mean is not the conditioned rate:

```
conditioned_rate = mean(metric) / mean(metric_applicable)
```

Oracle values for the current 40-scenario set:

| Metric | Raw mean | Applicable mean | Conditioned |
|--------|----------|-----------------|-------------|
| `namespace_match` | 0.975 | 0.975 | **1.000** |
| `chain_head_correct` | 0.950 | 0.950 | **1.000** |
| `chain_resolution_rate` | 0.950 | 0.950 | **1.000** |
| `propagation_edge_coverage` | 0.651 | 0.950 | **~0.685** |

`chain_head_correct` and `chain_resolution_rate` are at their ceiling on
applicable scenarios. `propagation_edge_coverage` is genuinely partial even
conditioned: ~0.685 is the oracle ceiling, not a perfect score.

Note: ~0.685 is the ceiling for oracle-style display-name chains, not for agents
in general. An agent that names real resource names rather than display names may
not trigger the same-stem collision and could score higher than the oracle on
this metric.

### Structural chain coverage ceiling (~0.685 conditioned)

Several scenarios have multiple groups whose filters share the same
`\b`-anchored stem (e.g. `shipping\b` for both a Deployment and a Service
group). Both resolve to the same synthesised literal and thus the same group id.
Consecutive resolved pairs in the oracle's propagation chain then include
self-loop edges (same id → same id) that are absent from the GT edge list.
Those pairs do not contribute to coverage, producing a genuine partial score
even on a perfect oracle run. This is a structural property of the dataset.

`chain_head_correct` passes cleanly despite the same-stem collision because the
first resolved id is the root-cause group id regardless of any downstream
self-loops.

### Namespace gating and scenario-102

`namespace_match` is a hard gate: a namespace mismatch blocks the entity from
claiming the GT group and zeros the headline reward.

`namespace_applicable` is determined by the GT group's `namespace` field:
`1.0` if the field is present and non-empty, `0.0` otherwise. Scenario-102's
root cause is a `Namespace` object; its GT group carries `namespace: null`
(Namespace objects do not belong to a namespace), so `namespace_applicable=0.0`
and `namespace_match=0.0` for that scenario. The condition is the absent field,
not an inferred cluster-scoped rule.

When `namespace_applicable=0.0`, `namespace_match` emits `0.0` — no vacuous
pass. The conditioned rate is recoverable as `mean(namespace_match) / mean(namespace_applicable)`.

### Alias resolution and kind matching

An agent may submit a different kind than the root-cause group if the two are in
the same alias class. `grade.py::_match_submitted_entity` accepts a submission,
in order:

1. **Exact same kind** — name must strict-fullmatch the root group's own filter.
2. **Authored alias sibling** — kind matches a sibling's kind AND name
   fullmatches *that sibling's* filter. This is ITBench-AA's own declaration of
   interchangeable representations; we honour it verbatim (see
   "Respecting the authored alias block").
3. **Workload equivalence** (see next section) — a controller/Pod/Service kind
   claims the root gated on the root's *own* name identity.

### Controller↔Pod workload equivalence

Pod / ReplicaSet / Deployment / StatefulSet / DaemonSet / Service are all
legitimate names for one logical workload, and telemetry does not reliably reveal
whether a config value (e.g. `LOCUST_USERS`) was "set on the Pod" vs "on the
owning controller". Forcing that distinction is impossibly strict and produced
false `reward=0` on scenario-1, where a workload-config fault is modelled as a
**Pod** root cause but naming the governing **Deployment** is the natural answer.

`_match_submitted_entity` therefore accepts a submitted workload-equivalent kind
against a workload-kind root **iff** the submitted name identifies the *root*
workload via `matching.name_matches_root_identity` — an anchored `re.fullmatch`
of the root group's own pattern, allowing only the trailing-hash-suffix
relaxation (`-.*`/`-.+` stripped). It never uses the lenient `name_matches`
(`re.search` + service-word widening), and a *sibling's* filter can never reach
the root through this path. Consequences:

- **Fair, not lenient:** a symmetric extension of the existing Pod↔Service alias
  to the owning controller, anchored to the *same* workload identity. It cannot
  admit a different workload (verified: `frontend-proxy`/Deployment and a foreign
  `adservice`/Deployment both score 0; unit tests in `test_grade.py`).
- **No `ground_truth.yaml` edits**, no owner lookup. Applied uniformly via the
  template `matching.py`/`grade.py` copied into every generated task.
- `resolve_chain_element` is root-aware with the *same* gate: an element matching
  the root identity resolves to the root group id (fixing `chain_head_correct`
  when a same-stem Service would otherwise win), while a victim/foreign name
  resolves to its own group, never the root head.
- Validated by `tests/test_oracle_end_to_end.py`: every scenario's oracle answer
  still scores 1.0 (no regression, nothing exceeds the oracle ceiling).

### Respecting the authored alias block

Scenario-1's `aliases` class lumps the root workload together with a **downstream
victim** (`frontend-proxy` Pod+Service). As a result an agent answering the
*symptom* `{name: frontend-proxy, kind: Service}` fullmatches the
`frontend-proxy-service-1` sibling filter and is scored a full match — an
arguably-wrong answer scored 1.0. This is **authored ground-truth behaviour**,
present before any change here.

We deliberately do **not** close this in the matcher. A gate that rejected it
(requiring the submitted name to also match the *root's* pattern) is structurally
indistinguishable from breaking a **legitimate** differently-named alias — e.g.
the `ALIAS_GT` case in `test_grade.py`, where a Service `root-svc` is a valid,
author-declared alias for pod `root-pod-*`. The grader cannot tell "legit
different-named alias" from "victim in alias class"; both are just a
differently-named sibling the author placed in the root's alias class. Overriding
that would make our grader **stricter than ITBench-AA declares**, i.e. deviate
from the benchmark. The correct fix lives in `ground_truth.yaml` (narrow the
alias class), which we do not edit. Flagged for upstream; see "Data quality
issues". The regression test `test_authored_alias_service_is_respected` locks in
that we keep honouring the authored block.

### Multi-root-cause scoring

All 40 current scenarios have exactly one root-cause group. The grader correctly
handles multi-root-cause ground truth (recall gate fires if any GT group is
uncovered), but this code path is exercised only by the unit test suite.

### Matching tolerances (no generation-time filter patches)

`ground_truth.yaml` is copied **verbatim** into each generated task
(`adapter.py._render_task`); there is no filter-rewriting step (no
`_KNOWN_FILTER_PATCHES`, no `patches_applied.json`). All matching tolerance lives
in the runtime matcher `tests/matching.py`, applied symmetrically to the oracle
(via `derive_matching_literal(strict=True)`) and to agent answers (via
`grade.py`):

- **Suffix relaxation** — a trailing owned-object hash wildcard (`-.*`/`-.+`) is
  stripped for anchored fullmatch, so the bare workload name matches a
  pod-suffixed filter. Strictly widens the anchored case only.
- **Workload equivalence** — see "Controller↔Pod workload equivalence"; gated on
  root identity.

These can only *widen* what matches and are validated to keep every oracle at
reward 1.0. The still-**open** service-name aliasing gap (`.*adservice` vs the
real name `ad`) is noted under "Data quality issues".

### Turn count calibration

AA publishes turn counts alongside scores for some models on itbench-aa. Sourced
from the AA blog post (cite when stable URL available):

| Model | Turns | Score |
|-------|-------|-------|
| GPT-5.5 (xhigh) | ~31 | ~46% |
| Gemma 4 31B (Reasoning) | ~58 | ~37% |
| Gemini 3.1 Pro Preview | ~83 | ~30% |

Opus 4.7 is not in the published AA turn-count data. `turn_count` in this
grader counts assistant messages in `/logs/agent/*.jsonl` transcripts. Cross-check
against these figures on the first real agent run to verify the instrumentation
is wired to the harness transcript.