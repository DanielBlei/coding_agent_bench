# SRE incident investigation

Alert(s) have fired, and you are the on-call SRE (Site Reliability Engineer) for this Kubernetes cluster.
Telemetry covering the incident window is captured under `/workspace/`, read-only.
Nothing is live: no kubectl, no cluster access, no way to change anything.
Whatever happened is already in the files.

**Your deliverable is the JSON answer at `/workspace/answer.json`.**
The investigation is not complete until it is written.

Identify the **root-cause entity or entities**: the object(s) whose own state or configuration explains the
alert, not ones merely showing symptoms of a problem that started elsewhere. An incident can have more than one
independent root cause.

## How to Interact

Start with `alerts/`. That is what paged you. Work backwards from what fired to what caused it.

| Path | Contents |
|---|---|
| `alerts/` | Per-minute Alertmanager snapshots (JSON) |
| `metrics/` | Prometheus samples, one TSV per pod/service |
| `k8s_events_raw.tsv`, `k8s_objects_raw.tsv` | Kubernetes events and objects |
| `otel_logs_raw.tsv`, `otel_traces_raw.tsv` | OpenTelemetry logs and traces |

Key details:

- The `*_raw.tsv` files are OpenTelemetry exports, not kubectl output. The real Kubernetes JSON (`kind`, `name`,
  `namespace`, spec) sits inside each row's `Body` column.
- In `metrics/`, `tags` is a Python dict string, not JSON. Values are raw counters, gauges and histogram buckets,
  not rates.
- All four `*_raw.tsv` files are quoted, may contain embedded newlines, and can run to hundreds of MB. Use `mlr`,
  `rg`, `jq` or `pandas.read_csv(path, sep="\t")`. Query them, do not read them whole.

## Output (Required)

Shape of `/workspace/answer.json`:

```json
{
  "schema_version": "1.0",
  "root_causes": [
    {"name": "<exact object name, not an inferred label>", "kind": "<Kubernetes kind>", "namespace": "<omit for cluster-scoped>"}
  ],
  "reasoning": "<evidence this is the origin, not a symptom, and why alternatives are ruled out>",
  "propagation_chain": ["<root-cause-name>", "<intermediate-name>", "<symptom-name>"],
  "recommended_actions": ["<remediation step>"]
}
```

- `root_causes` is an array: one entry per independent root cause. List only entities you have evidence for.
- `name`: the object's real name as it appears in the Kubernetes objects, not an inferred or shortened label.
- `propagation_chain`: **entity names only, one per hop**, ordered root cause -> ... -> symptom, not sentences.
  Use a single-element list if nothing propagated.
- Before closing out, confirm `/workspace/answer.json` exists and parses as JSON.
