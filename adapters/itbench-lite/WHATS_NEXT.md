# itbench-lite: what's next — closing the deterministic-vs-judge gap

Reoriented from an earlier publishing-first draft. Publishing (Harbor
upstream, M3) is real but *later* — see the bottom section for a pointer.
The actual next work is **validation**: our grader is deterministic
(`kind`/`name`/`namespace` regex match), IBM's is a hybrid LLM-judge
(`gpt-4-turbo`), and nobody has yet measured how often they'd disagree. That
measurement — not publish-readiness — is the priority.

Full technical detail and citations for everything below live in
[`note-sre-leaderboard-itbench-strategy-milestones.md`](file:///home/dblei/Development/knowledge-base/vault/team-ace/initiatives/sre-leaderboard/notes/note-sre-leaderboard-itbench-strategy-milestones.md)
(2026-08-27) — this doc doesn't re-derive that analysis, it turns it into an
ordered, actionable list: **each gap below names what differs, why, how to
close it, how much effort, and who the real work traces back to** (so we
credit IBM's methodology/data instead of presenting a diff or a schema
change as something we invented). Work these one at a time — they don't
have to happen in one sitting.

## The headline easy win: we can run the diff study with zero new agent runs

IBM already published **judged agent outputs on the exact same 35
scenarios**: [`ibm-research/ITBench-Trajectories`](https://huggingface.co/datasets/ibm-research/ITBench-Trajectories)
— 105 runs (35 scenarios × 3 repeats) of **GPT-OSS-120B**, each with a
`judge_output.json` containing the judge's per-entity match decisions and
P/R/F1. That means the deterministic-vs-judge study (milestone doc's **M1**)
doesn't require running our own agent *or* our own judge — it requires:

1. Pull `ITBench-Trajectories`, extract each run's predicted entity list.
2. Run our existing `grade.py`/`matching.py` against those predictions (same
   `ground_truth.yaml` files we already have).
3. Diff our verdicts against IBM's `judge_output.json` verdicts on the same
   105 runs.

No compute budget, no API key, no new infra — this is code + data ingestion
against artifacts IBM has already produced and published. **This should be
the very next concrete task**, and it's what most of the gaps below get
tested against rather than argued about in the abstract.

**Credit:** the ability to do this at zero cost is entirely because of IBM's
own published trajectories dataset — any findings memo from this study
should open by citing [`ITBench-Trajectories`](https://huggingface.co/datasets/ibm-research/ITBench-Trajectories)
and [`ITBench-Evaluations`](https://github.com/itbench-hub/ITBench-Evaluations)
(the grader that produced `judge_output.json`) as the source of the
comparison data, not describe the disagreement numbers as something we
measured from scratch.

## Gap register

Ordered by recommended sequence — earlier items unblock or cheapen later
ones. Reference by number ("let's do Gap 3") to work through incrementally.

### Gap 1 — Answer schema: flat single-cause vs. IBM's `entities[]` graph

- **Differs:** our `answer.json` is `{root_cause, kind, namespace, reasoning,
  propagation_chain[]}`; IBM's contract is a graph — `entities[]` (each with
  `namespace/Kind/name`, `contributing_factor`, `reasoning`), `propagations[]`
  (edges with `condition`/`effect`), `alerts_explained[]`.
- **Why we diverged:** v1 simplification, written before this adapter was
  compared against the reference agent's actual output contract.
- **How to close:** switch the schema to `entities[]` / `namespace/Kind/name`.
  Milestone doc calls this "small, unblocks everything" (M0) — it's what
  makes our outputs runnable by *either* grader (ours or IBM's judge) at
  once, which is the precondition for the diff study above being fully
  apples-to-apples at entity-list granularity (not just single-answer).
- **Effort:** low-medium. It's a schema/parsing change in `adapter.py` +
  `grade.py`, not new logic — the underlying match/compute code stays.
- **Credit:** the target schema is IBM's own reference agent's `FINAL OUTPUT
  FORMAT` contract
  ([`sre_react_shell_investigation.md`](https://github.com/itbench-hub/ITBench-CISO-SRE-FinOps-Agent/blob/main/zero/zero-config/prompts/sre_react_shell_investigation.md)) —
  cite that prompt directly as the source of the shape, since we're
  conforming to it, not designing it.

### Gap 2 — Entity-match aliases blindness (the leading disagreement hypothesis)

- **Differs:** IBM's judge uses the ground truth's `aliases` field (plus
  kind hints) to decide entity equivalence. Our matcher **deliberately never
  reads `aliases`** — it only regexes against `filter[]`.
- **Why we diverged:** not a considered tradeoff so much as an oversight —
  the `aliases` field wasn't examined when the matcher was built.
- **How to close:** **don't guess — let the headline diff study answer this
  first.** The milestone doc's own leading hypothesis is that most
  deterministic-vs-judge disagreement will turn out to be alias-driven, which
  would make this "a bounded, countable disagreement, not an open-ended
  one." Once the diff quantifies it, decide whether to consult `aliases` in
  `matching.py` (risk: over-crediting matches the judge wouldn't have
  accepted either — gate any change on what the diff actually shows).
- **Effort:** the diff itself is covered by the headline win above (already
  free). The matcher change, if warranted, is small — one more widening
  rule in `matching.py`, same shape as the existing suffix/service-token/
  fnmatch widenings.
- **Credit:** the alias-equivalence *rule* itself is IBM's — it's defined in
  [`ITBench-Evaluations`](https://github.com/itbench-hub/ITBench-Evaluations)'
  judge logic, not something to reverse-engineer and claim as original. If
  implemented, the commit/README note should say "adopts IBM's
  alias-equivalence rule from `ITBench-Evaluations`," not "improved entity
  matching."

### Gap 3 — Propagation/proximity signals: computed but unscored

- **Differs:** we already compute `chain_proximity` and
  `propagation_chain_coverage` in `grade.py` — they're just informational,
  never gating `reward`. IBM scores the equivalent (`propagation`,
  `hop-proximity`) as part of the judged metric set.
- **Why we diverged:** original v1 design chose a strict
  `kind ∧ name ∧ namespace` AND-gate for simplicity.
- **How to close:** literally a wiring change — decide these count toward a
  score, adopt IBM's metric names (P/R/F1, entity@k, pass@1) for the
  reported fields. No new computation needed, the numbers already exist in
  every `reward.json`.
- **Effort:** low — this is the cheapest real improvement on the list.
- **Credit:** metric *names* and definitions come from `ITBench-Evaluations`'
  [metrics section](https://github.com/itbench-hub/ITBench-Evaluations/blob/main/README.md#metrics-covered) —
  reuse their vocabulary rather than inventing adapter-specific metric
  names, so results stay comparable at a glance.

### Gap 4 — Reasoning quality unscored

- **Differs:** IBM's judge grades reasoning 0/0.5/1 (right resource + good
  explanation = 1, right resource + vague explanation = 0.5, wrong = 0). We
  only check `reasoning_present` — a ≥20-character length proxy, never
  gating reward.
- **Why we diverged:** avoiding an LLM call in the core deterministic
  verifier (deliberate, still a reasonable default).
- **How to close:** the milestone doc's proposed variant — keep the Harbor
  verifier fully deterministic, add an **optional judge-on-top pass** as a
  second, clearly-labelled, non-deterministic layer that only adds the
  reasoning grade (and can double-check fuzzy entity matches). Don't fold it
  into `reward`.
- **Effort:** medium — this is the one item here that requires actual judge
  infrastructure (model calls, cost, non-determinism to manage). Sequence it
  after Gaps 1–3, since the diff study (Gap 2) may reduce how much this
  matters in practice.
- **Credit:** the 0/0.5/1 rubric is IBM's; if built, cite
  [`ITBench-Evaluations`](https://github.com/itbench-hub/ITBench-Evaluations)
  as the rubric source rather than presenting a new grading scale.

### Gap 5 — Multi-cause incidents and alert-coverage dropped

- **Differs:** IBM's schema supports multiple independent root causes
  (`contributing_factor: true` entities, gated by an "irreducibility test"
  so a symptom isn't double-counted as a cause) and requires every alert be
  explained (`alerts_explained[]`). We score exactly one root-cause group
  per scenario and don't check alert coverage at all.
- **Why we diverged:** v1 scope simplification.
- **How to close:** first, a cheap check before any implementation work —
  script a count of how many of the 35 scenarios' ground truth actually has
  more than one `root_cause: true` group. If it's rare or zero in this
  35-scenario set, this gap is more theoretical than practical right now
  and can stay deferred without loss. If it's common, this becomes a real
  scoring-fidelity gap worth closing via Gap 1's schema work.
- **Effort:** the check is trivial (minutes). Full support is medium-high
  and depends on what the check finds.
- **Credit:** the "irreducibility test" concept (don't mark both a cause and
  its downstream symptom as contributing) is from IBM's reference agent
  prompt — cite it directly if this logic is ported.

### Gap 6 — Tooling confound (raw CLI vs. MCP tools) — fairness note, not a grading gap

- **Differs:** IBM's reference agent gets a 10-tool MCP server
  (`offline_incident_analysis` — pre-aggregated alerts, anomaly detection,
  topology queries, etc.). Our agent gets generic CLI (`jq`/`mlr`/`rg`/
  `pandas`) over raw multi-hundred-MB TSVs and must write its own parsing.
- **Why we diverged:** deliberate — no MCP server to version/maintain, works
  with any container-capable agent, keeps the task self-contained.
- **How to close:** this doesn't get "closed" the way the grading gaps do —
  it's a scoring-relevant methodological choice to keep disclosing (already
  is, in the README), not a bug. Only action item: **don't compare our
  scores to IBM's published numbers without footnoting this**, since our
  task is strictly harder.
- **Effort:** n/a (disclosure, not implementation) — unless the M5 "later"
  track ever wants a toggleable tool layer for a fairer head-to-head.
- **Credit:** n/a — this is our own deliberate divergence, not a port of
  IBM's work.

## Sequencing summary

1. Gap 1 (schema → `entities[]`) — unblocks full parity comparisons.
2. Headline win (run `grade.py` against `ITBench-Trajectories`) — the actual
   M1 deliverable, doable in parallel with or right after Gap 1.
3. Gap 2 (aliases) — decide only after the diff data exists.
4. Gap 3 (score existing propagation/proximity signals) — cheap, do anytime.
5. Gap 5's quick multi-cause prevalence check — cheap, do anytime, informs
   whether Gap 5 is worth real investment.
6. Gap 4 (judge-on-top layer) — after the above, since it's the highest-effort
   item and Gap 2's findings may change how much it's needed.
7. Gap 6 — ongoing disclosure hygiene, not a task with an end state.

## Giving credit — where each future write-up should point

Any findings memo, README update, or commit message coming out of this work
should cite the specific IBM artifact it's built on, not describe the result
as originating in-house:

- **Diff-study data & grading rubric:** [`ITBench-Evaluations`](https://github.com/itbench-hub/ITBench-Evaluations) —
  the LLM-judge toolkit that defines entity-equivalence and the 0/0.5/1
  reasoning rubric.
- **Pre-judged comparison runs:** [`ibm-research/ITBench-Trajectories`](https://huggingface.co/datasets/ibm-research/ITBench-Trajectories) —
  the 105 GPT-OSS-120B runs the headline win diffs against.
- **Output schema being adopted:** [`ITBench-CISO-SRE-FinOps-Agent`](https://github.com/itbench-hub/ITBench-CISO-SRE-FinOps-Agent)'s
  system prompt — the `entities[]`/`propagations[]`/`alerts_explained[]`
  contract.
- **Ground truth & ITBench-Lite dataset itself:** [`ibm-research/ITBench-Lite`](https://huggingface.co/datasets/ibm-research/ITBench-Lite).
- **Paper/background:** ITBench — [arXiv:2502.05352](https://arxiv.org/abs/2502.05352).

## Later: publishing (M3)

Not the current focus. When this work resumes, the full publish-blocker
checklist (LICENSE, `parity_experiment.json`, dataset layout, mount-based
data path, non-hermetic verifier, etc.) already lives in
[`note-sre-leaderboard-itbench-strategy-milestones.md`](file:///home/dblei/Development/knowledge-base/vault/team-ace/initiatives/sre-leaderboard/notes/note-sre-leaderboard-itbench-strategy-milestones.md)'s
"Pushing to Harbor" section — no need to re-derive it here. Notably,
`parity_experiment.json`
*is* the output of the headline win + Gap 1–5 work above, so this section
naturally becomes actionable once those land, not before.
