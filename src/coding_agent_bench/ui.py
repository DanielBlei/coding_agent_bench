"""UI module for job submission and viewing."""

import json
import html

from coding_agent_bench.models import MODEL_REGISTRY
from coding_agent_bench.nebius_utils import RESOURCE_CONFIGS

NEBIUS_PREFIX = "nebius-"


def build_submit_form_html(
    models: list[str],
    agents: list[str],
    nebius_configs: list[str],
    nebius_enabled: bool,
) -> str:
    """Build the HTML for the job submission form."""

    # Define basic fields (always visible)
    basic_fields = _build_basic_fields_html(models, agents, nebius_configs, nebius_enabled)

    # Define optional/advanced fields
    advanced_fields = _build_advanced_fields_html(nebius_configs, nebius_enabled)
    validation_table = _build_model_gpu_validation_table_html()

    return f"""
<div id="submit-job-section" style="margin-bottom: 2rem; padding: 1rem; border: 1px solid #ddd; border-radius: 8px; background: #fafafa;">
    <h2 style="margin-top: 0;">Submit New Job</h2>
    <form id="job-form" onsubmit="return submitJob(event)">
        <!-- Basic Fields (always visible) -->
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;">
            {basic_fields}
        </div>

        <!-- Advanced Options Toggle -->
        <div style="margin-top: 1rem;">
            <button type="button" id="advanced-toggle" onclick="toggleAdvanced()" style="background: none; border: none; color: #0066cc; cursor: pointer; text-decoration: underline; padding: 0;">
                Show Advanced Options ▾
            </button>
        </div>

        <!-- Advanced Fields (hidden by default) -->
        <div id="advanced-fields" style="display: none; margin-top: 1rem; padding-top: 1rem; border-top: 1px solid #eee;">
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;">
                {advanced_fields}
            </div>
        </div>

        <!-- Submit Button -->
        <div style="margin-top: 1.5rem;">
            <button type="submit" id="submit-btn" style="background: #0066cc; color: white; border: none; padding: 0.75rem 1.5rem; border-radius: 4px; cursor: pointer; font-size: 1rem;">
                Submit Job
            </button>
            <span id="submit-status" style="margin-left: 1rem; font-style: italic;"></span>
        </div>

        {validation_table}
    </form>
</div>

<script>
const MODEL_OPTIONS = {json.dumps(models)};
const AGENT_OPTIONS = {json.dumps(agents)};
const NEBIUS_CONFIGS = {json.dumps(nebius_configs)};
const NEBIUS_ENABLED = {json.dumps(nebius_enabled)};
const NEBIUS_PREFIX = '{NEBIUS_PREFIX}';

function toggleAdvanced() {{
    const adv = document.getElementById('advanced-fields');
    const btn = document.getElementById('advanced-toggle');
    if (adv.style.display === 'none') {{
        adv.style.display = 'block';
        btn.textContent = 'Hide Advanced Options ▴';
    }} else {{
        adv.style.display = 'none';
        btn.textContent = 'Show Advanced Options ▾';
    }}
}}

function validateForm() {{
    const errors = [];
    const agent = document.getElementById('agent').value;
    const dataset = document.getElementById('dataset').value.trim();
    const modelName = document.getElementById('model_name').value;
    const serverUrl = document.getElementById('server_url').value.trim();

    if (!agent) errors.push('Agent is required');
    if (!dataset) errors.push('Dataset is required');
    if (!modelName) errors.push('Model name is required');
    if (!serverUrl) errors.push('Server URL is required');

    // Validate server_url format
    if (serverUrl) {{
        if (!serverUrl.startsWith(NEBIUS_PREFIX) && !serverUrl.match(/^https?:\\/\\/[^\\s]+$/)) {{
            errors.push('Server URL must be a valid URL (http:// or https://)');
        }}
        if (serverUrl.startsWith(NEBIUS_PREFIX)) {{
            const config = serverUrl.substring(NEBIUS_PREFIX.length);
            if (NEBIUS_ENABLED && NEBIUS_CONFIGS.length > 0 && !NEBIUS_CONFIGS.includes(config)) {{
                errors.push('Unknown nebius resource config. Choose from: ' + NEBIUS_CONFIGS.join(', '));
            }}
            if (!NEBIUS_ENABLED) {{
                errors.push('Nebius is not enabled on this server');
            }}
        }}
    }}

    // Validate numeric fields
    const nConcurrent = document.getElementById('n_concurrent').value;
    if (nConcurrent && (isNaN(nConcurrent) || parseInt(nConcurrent) < 1)) {{
        errors.push('n_concurrent must be a positive integer');
    }}
    const nTasks = document.getElementById('n_tasks').value;
    if (nTasks && (isNaN(nTasks) || parseInt(nTasks) < 1)) {{
        errors.push('n_tasks must be a positive integer');
    }}
    const modelMaxLen = document.getElementById('model_max_len').value;
    if (modelMaxLen && (isNaN(modelMaxLen) || parseInt(modelMaxLen) < 1)) {{
        errors.push('model_max_len must be a positive integer');
    }}
    const timeoutMultiplier = document.getElementById('agent_timeout_multiplier').value.trim();
    if (timeoutMultiplier && (isNaN(timeoutMultiplier) || parseFloat(timeoutMultiplier) <= 0)) {{
        errors.push('agent_timeout_multiplier must be a positive number');
    }}
    const maxRetries = document.getElementById('max_retries').value.trim();
    if (maxRetries && (isNaN(maxRetries) || parseInt(maxRetries) < 0)) {{
        errors.push('max_retries must be a non-negative integer');
    }}

    // Validate KEY=VALUE textareas (agent_env, agent_kwargs, env_kwargs, harbor_envs)
    for (const [fieldId, label] of [
        ['agent_env', 'Agent env'],
        ['agent_kwargs', 'Agent kwargs'],
        ['env_kwargs', 'Env kwargs'],
        ['harbor_envs', 'Harbor envs'],
    ]) {{
        for (const line of parseKeyValueLines(fieldId)) {{
            if (!line.includes('=')) {{
                errors.push(label + ' entries must be in KEY=VALUE format: ' + line);
            }}
        }}
    }}

    if (errors.length > 0) {{
        alert('Validation errors:\\n' + errors.join('\\n'));
        return false;
    }}
    return true;
}}

function sanitizeJobNamePart(value) {{
    return value.toLowerCase().replace(/[^a-z0-9_-]/g, '-');
}}

function updateJobName() {{
    const modelName = document.getElementById('model_name').value;
    const dataset = document.getElementById('dataset').value.trim();
    const agent = document.getElementById('agent').value;
    const jobName = [modelName, dataset, agent].map(sanitizeJobNamePart).join('-');
    document.getElementById('job_name').value = jobName;
}}

function parseKeyValueLines(fieldId) {{
    // Values may themselves contain commas (e.g. JSON), so only newlines separate entries.
    return document.getElementById(fieldId).value
        .split(/\\r?\\n/)
        .map((line) => line.trim())
        .filter(Boolean);
}}

function parseListField(fieldId) {{
    // Hosts and error-type names never contain commas, so accept both separators.
    return document.getElementById(fieldId).value
        .split(/[\\n,]/)
        .map((item) => item.trim())
        .filter(Boolean);
}}

async function submitJob(event) {{
    event.preventDefault();
    updateJobName();
    if (!validateForm()) return false;

    const btn = document.getElementById('submit-btn');
    const status = document.getElementById('submit-status');
    btn.disabled = true;
    btn.textContent = 'Submitting...';
    status.textContent = '';
    status.style.color = '#666';

    while (true) {{
        const lookupPromise = concurrencyLookupPromise;
        await lookupPromise;
        if (lookupPromise === concurrencyLookupPromise) break;
    }}

    const formData = {{
        job_name: document.getElementById('job_name').value.trim(),
        agent: document.getElementById('agent').value,
        dataset: document.getElementById('dataset').value.trim(),
        model_name: document.getElementById('model_name').value,
        server_url: document.getElementById('server_url').value.trim(),
        n_concurrent: parseInt(document.getElementById('n_concurrent').value) || 1,
    }};

    // Advanced fields
    const datasetPattern = document.getElementById('dataset_pattern').value.trim();
    if (datasetPattern) formData.dataset_pattern = datasetPattern;

    const nTasks = document.getElementById('n_tasks').value.trim();
    if (nTasks) formData.n_tasks = parseInt(nTasks);

    const modelMaxLen = document.getElementById('model_max_len').value.trim();
    if (modelMaxLen) formData.model_max_len = parseInt(modelMaxLen);

    const beforeScript = document.getElementById('before_script').value.trim();
    if (beforeScript) formData.before_script = beforeScript;

    const agentVersion = document.getElementById('agent_version').value.trim();
    if (agentVersion) formData.agent_version = agentVersion;

    const skills = document.getElementById('skills').value
        .split(',')
        .map((skill) => skill.trim())
        .filter(Boolean);
    if (skills.length) formData.skills = skills;

    const agentEnv = parseKeyValueLines('agent_env');
    if (agentEnv.length) formData.agent_env = agentEnv;

    const agentKwargs = parseKeyValueLines('agent_kwargs');
    if (agentKwargs.length) formData.agent_kwargs = agentKwargs;

    const envKwargs = parseKeyValueLines('env_kwargs');
    if (envKwargs.length) formData.env_kwargs = envKwargs;

    const harborEnvs = parseKeyValueLines('harbor_envs');
    if (harborEnvs.length) formData.harbor_envs = harborEnvs;

    const retryInclude = parseListField('retry_include');
    if (retryInclude.length) formData.retry_include = retryInclude;

    const allowAgentHost = parseListField('allow_agent_host');
    if (allowAgentHost.length) formData.allow_agent_host = allowAgentHost;

    const timeoutMultiplier = document.getElementById('agent_timeout_multiplier').value.trim();
    if (timeoutMultiplier) formData.agent_timeout_multiplier = parseFloat(timeoutMultiplier);

    const maxRetries = document.getElementById('max_retries').value.trim();
    if (maxRetries !== '') formData.max_retries = parseInt(maxRetries);

    try {{
        const apiKey = localStorage.getItem('coding_agent_bench_api_key');
        const response = await fetch('/jobs', {{
            method: 'POST',
            headers: {{
                'Content-Type': 'application/json',
                'X-API-Key': apiKey || '',
            }},
            body: JSON.stringify(formData),
        }});

        const data = await response.json();

        if (!response.ok) {{
            throw new Error(data.detail || `HTTP ${{response.status}}`);
        }}

        status.textContent = `Job created: ${{data.job_id}}`;
        status.style.color = 'green';
        document.getElementById('job-form').reset();

        // Refresh job list after a short delay
        setTimeout(() => location.reload(), 1500);
    }} catch (err) {{
        status.textContent = `Error: ${{err.message}}`;
        status.style.color = 'red';
        btn.disabled = false;
        btn.textContent = 'Submit Job';
    }}
}}

function checkApiKey() {{
    const apiKey = localStorage.getItem('coding_agent_bench_api_key');
    const status = document.getElementById('submit-status');
    if (!apiKey) {{
        status.textContent = '⚠ API key not set. Set it in the header above first.';
        status.style.color = '#cc6600';
    }}
}}

let concurrencyLookupId = 0;
let concurrencyLookupPromise = Promise.resolve();
let lookedUpConcurrency = null;

async function updateMaxConcurrency() {{
    const modelName = document.getElementById('model_name').value;
    const serverUrl = document.getElementById('server_url').value.trim();
    const concurrencyInput = document.getElementById('n_concurrent');
    const warning = document.getElementById('n-concurrent-warning');
    const lookupId = ++concurrencyLookupId;

    warning.textContent = '';
    if (concurrencyInput.value === lookedUpConcurrency) concurrencyInput.value = '';
    lookedUpConcurrency = null;

    if (!modelName || !serverUrl.toLowerCase().startsWith(NEBIUS_PREFIX)) return;

    const gpuConfig = serverUrl.substring(NEBIUS_PREFIX.length).toLowerCase();
    if (!gpuConfig) return;

    try {{
        const params = new URLSearchParams({{ model_name: modelName, gpu_config: gpuConfig }});
        const response = await fetch(`/api/model-max-concurrency?${{params}}`);

        if (lookupId !== concurrencyLookupId) return;

        if (response.status === 404) {{
            warning.textContent = 'No validated concurrency is available for this model and hardware. You can still submit the job.';
            return;
        }}
        if (!response.ok) throw new Error(`HTTP ${{response.status}}`);

        const data = await response.json();
        if (Number.isInteger(data.max_concurrency) && data.max_concurrency > 0) {{
            concurrencyInput.value = data.max_concurrency;
            lookedUpConcurrency = String(data.max_concurrency);
        }}
    }} catch (error) {{
        if (lookupId === concurrencyLookupId) console.warn('Could not fetch max concurrency:', error);
    }}
}}

function startMaxConcurrencyLookup() {{
    concurrencyLookupPromise = updateMaxConcurrency();
}}

document.getElementById('model_name').addEventListener('change', startMaxConcurrencyLookup);
document.getElementById('server_url').addEventListener('change', startMaxConcurrencyLookup);
document.getElementById('model_name').addEventListener('change', updateJobName);
document.getElementById('agent').addEventListener('change', updateJobName);
document.getElementById('dataset').addEventListener('input', updateJobName);

updateJobName();
checkApiKey();
</script>
"""


def _build_basic_fields_html(
    models: list[str],
    agents: list[str],
    nebius_configs: list[str],
    nebius_enabled: bool,
) -> str:
    """Build HTML for the basic (always visible) form fields."""
    model_options = "".join(
        f'<option value="{html.escape(m)}">{html.escape(m)}</option>' for m in models
    )
    agent_options = "".join(
        f'<option value="{html.escape(a)}">{html.escape(a)}</option>' for a in agents
    )
    nebius_config_options = "".join(
        f'<option value="{html.escape(NEBIUS_PREFIX + config)}">'
        for config in nebius_configs
    )

    nebius_help = ""
    if nebius_enabled:
        nebius_help = '<br><small style="color: #666;">Or use nebius-&lt;config&gt; (e.g., nebius-h200)</small>'
    else:
        nebius_help = '<br><small style="color: #999;">Nebius instances not enabled</small>'

    return f"""
        <div>
            <label for="job_name" style="display: block; font-weight: bold; margin-bottom: 0.25rem;">Job Name</label>
            <input type="text" id="job_name" name="job_name" readonly
                   style="width: 100%; padding: 0.5rem; border: 1px solid #ccc; border-radius: 4px; box-sizing: border-box;"
                   placeholder="Generated from model, benchmark, and agent">
        </div>
        <div>
            <label for="agent" style="display: block; font-weight: bold; margin-bottom: 0.25rem;">Agent *</label>
            <select id="agent" name="agent" required
                    style="width: 100%; padding: 0.5rem; border: 1px solid #ccc; border-radius: 4px; box-sizing: border-box;">
                {agent_options}
            </select>
        </div>
        <div>
            <label for="dataset" style="display: block; font-weight: bold; margin-bottom: 0.25rem;">Dataset *</label>
            <input type="text" id="dataset" name="dataset" required
                   style="width: 100%; padding: 0.5rem; border: 1px solid #ccc; border-radius: 4px; box-sizing: border-box;"
                   placeholder="e.g., humaneval">
        </div>
        <div>
            <label for="model_name" style="display: block; font-weight: bold; margin-bottom: 0.25rem;">Model *</label>
            <select id="model_name" name="model_name" required
                    style="width: 100%; padding: 0.5rem; border: 1px solid #ccc; border-radius: 4px; box-sizing: border-box;">
                {model_options}
            </select>
        </div>
        <div>
            <label for="server_url" style="display: block; font-weight: bold; margin-bottom: 0.25rem;">Server URL *</label>
            <input type="text" id="server_url" name="server_url" required
                   list="nebius-config-options"
                   style="width: 100%; padding: 0.5rem; border: 1px solid #ccc; border-radius: 4px; box-sizing: border-box;"
                   placeholder="http://localhost:8000">
            <datalist id="nebius-config-options">{nebius_config_options}</datalist>
            {nebius_help}
        </div>
        <div>
            <label for="n_concurrent" style="display: block; font-weight: bold; margin-bottom: 0.25rem;">Concurrent Tasks</label>
            <input type="number" id="n_concurrent" name="n_concurrent" value="1" min="1"
                   style="width: 100%; padding: 0.5rem; border: 1px solid #ccc; border-radius: 4px; box-sizing: border-box;">
            <small id="n-concurrent-warning" style="display: block; color: #cc6600; margin-top: 0.25rem;"></small>
        </div>
"""


def _build_advanced_fields_html(nebius_configs: list[str], nebius_enabled: bool) -> str:
    """Build HTML for the optional/advanced form fields."""
    return """
        <div>
            <label for="dataset_pattern" style="display: block; font-weight: bold; margin-bottom: 0.25rem;">Dataset Pattern</label>
            <input type="text" id="dataset_pattern" name="dataset_pattern"
                   style="width: 100%; padding: 0.5rem; border: 1px solid #ccc; border-radius: 4px; box-sizing: border-box;"
                   placeholder="e.g., *hard*">
        </div>
        <div>
            <label for="n_tasks" style="display: block; font-weight: bold; margin-bottom: 0.25rem;">Total Tasks</label>
            <input type="number" id="n_tasks" name="n_tasks" min="1"
                   style="width: 100%; padding: 0.5rem; border: 1px solid #ccc; border-radius: 4px; box-sizing: border-box;"
                   placeholder="Leave empty for all">
        </div>
        <div>
            <label for="model_max_len" style="display: block; font-weight: bold; margin-bottom: 0.25rem;">Max Context Length</label>
            <input type="number" id="model_max_len" name="model_max_len" min="1"
                   style="width: 100%; padding: 0.5rem; border: 1px solid #ccc; border-radius: 4px; box-sizing: border-box;"
                   placeholder="Leave empty for default">
        </div>
        <div>
            <label for="agent_version" style="display: block; font-weight: bold; margin-bottom: 0.25rem;">Agent Version</label>
            <input type="text" id="agent_version" name="agent_version"
                   style="width: 100%; padding: 0.5rem; border: 1px solid #ccc; border-radius: 4px; box-sizing: border-box;"
                   placeholder="Leave empty for default">
        </div>
        <div style="grid-column: 1 / -1;">
            <label for="skills" style="display: block; font-weight: bold; margin-bottom: 0.25rem;">Skills</label>
            <textarea id="skills" name="skills" rows="2"
                      style="width: 100%; padding: 0.5rem; border: 1px solid #ccc; border-radius: 4px; box-sizing: border-box; font-family: monospace;"
                      placeholder="Comma-separated Git sources, e.g. org/repo@main"></textarea>
            <small style="color: #666;">Provide Git sources separated by commas.</small>
        </div>
        <div style="grid-column: 1 / -1;">
            <label for="before_script" style="display: block; font-weight: bold; margin-bottom: 0.25rem;">Before Script</label>
            <textarea id="before_script" name="before_script" rows="3"
                      style="width: 100%; padding: 0.5rem; border: 1px solid #ccc; border-radius: 4px; box-sizing: border-box; font-family: monospace;"
                      placeholder="Commands to run before execution..."></textarea>
        </div>
        <div>
            <label for="agent_timeout_multiplier" style="display: block; font-weight: bold; margin-bottom: 0.25rem;">Agent Timeout Multiplier</label>
            <input type="number" id="agent_timeout_multiplier" name="agent_timeout_multiplier" min="0.1" step="0.5"
                   style="width: 100%; padding: 0.5rem; border: 1px solid #ccc; border-radius: 4px; box-sizing: border-box;"
                   placeholder="Leave empty for default">
        </div>
        <div>
            <label for="max_retries" style="display: block; font-weight: bold; margin-bottom: 0.25rem;">Max Retries</label>
            <input type="number" id="max_retries" name="max_retries" min="0"
                   style="width: 100%; padding: 0.5rem; border: 1px solid #ccc; border-radius: 4px; box-sizing: border-box;"
                   placeholder="Leave empty for default (1)">
        </div>
        <div>
            <label for="retry_include" style="display: block; font-weight: bold; margin-bottom: 0.25rem;">Retry Error Types</label>
            <textarea id="retry_include" name="retry_include" rows="2"
                      style="width: 100%; padding: 0.5rem; border: 1px solid #ccc; border-radius: 4px; box-sizing: border-box; font-family: monospace;"
                      placeholder="One per line or comma-separated, e.g. AgentTimeoutError"></textarea>
            <small style="color: #666;">Error types to retry (Harbor's --retry-include). Leave empty for defaults.</small>
        </div>
        <div>
            <label for="allow_agent_host" style="display: block; font-weight: bold; margin-bottom: 0.25rem;">Allow Agent Hosts</label>
            <textarea id="allow_agent_host" name="allow_agent_host" rows="2"
                      style="width: 100%; padding: 0.5rem; border: 1px solid #ccc; border-radius: 4px; box-sizing: border-box; font-family: monospace;"
                      placeholder="One per line or comma-separated, e.g. api.example.com"></textarea>
            <small style="color: #666;">Extra hostnames/IPs/CIDRs for the agent-phase network allowlist (Harbor's --allow-agent-host).</small>
        </div>
        <div style="grid-column: 1 / -1;">
            <label for="agent_env" style="display: block; font-weight: bold; margin-bottom: 0.25rem;">Agent Env Vars (--ae)</label>
            <textarea id="agent_env" name="agent_env" rows="3"
                      style="width: 100%; padding: 0.5rem; border: 1px solid #ccc; border-radius: 4px; box-sizing: border-box; font-family: monospace;"
                      placeholder="One KEY=VALUE per line; overrides agent-config values, e.g.&#10;ANTHROPIC_BASE_URL=https://gateway.example&#10;CLAUDE_CODE_MAX_OUTPUT_TOKENS=64000"></textarea>
            <small style="color: #666;">Passed to Harbor's --ae. Values are stored with the job and visible in its command.</small>
        </div>
        <div style="grid-column: 1 / -1;">
            <label for="agent_kwargs" style="display: block; font-weight: bold; margin-bottom: 0.25rem;">Agent Kwargs (--ak)</label>
            <textarea id="agent_kwargs" name="agent_kwargs" rows="2"
                      style="width: 100%; padding: 0.5rem; border: 1px solid #ccc; border-radius: 4px; box-sizing: border-box; font-family: monospace;"
                      placeholder="One key=value per line, e.g. reasoning_effort=high"></textarea>
            <small style="color: #666;">Passed to Harbor's --ak; overrides agent-config kwargs on key collision.</small>
        </div>
        <div style="grid-column: 1 / -1;">
            <label for="env_kwargs" style="display: block; font-weight: bold; margin-bottom: 0.25rem;">Environment Kwargs (--ek)</label>
            <textarea id="env_kwargs" name="env_kwargs" rows="2"
                      style="width: 100%; padding: 0.5rem; border: 1px solid #ccc; border-radius: 4px; box-sizing: border-box; font-family: monospace;"
                      placeholder="One key=value per line, e.g. namespace=my-namespace"></textarea>
            <small style="color: #666;">Passed to Harbor's --ek (environment kwargs).</small>
        </div>
        <div style="grid-column: 1 / -1;">
            <label for="harbor_envs" style="display: block; font-weight: bold; margin-bottom: 0.25rem;">Harbor Process Env Vars (--envs)</label>
            <textarea id="harbor_envs" name="harbor_envs" rows="2"
                      style="width: 100%; padding: 0.5rem; border: 1px solid #ccc; border-radius: 4px; box-sizing: border-box; font-family: monospace;"
                      placeholder="One KEY=VALUE per line, e.g. DATA_ROOT=/path/to/data"></textarea>
            <small style="color: #666;">Environment variables for the harbor process itself. Values are stored with the job and visible in its command.</small>
        </div>
    """


def _build_model_gpu_validation_table_html() -> str:
    """Build a table showing validated model/GPU concurrency combinations."""
    gpu_configs = list(RESOURCE_CONFIGS)
    header_cells = "".join(
        f'<th scope="col">{html.escape(config.name)}</th>' for config in gpu_configs
    )

    rows = []
    for model_name, model_config in MODEL_REGISTRY.items():
        cells = []
        for gpu_config in gpu_configs:
            concurrency = gpu_config.model_concurrency.get(model_config.name)
            value = str(concurrency) if concurrency is not None else "&mdash;"
            cells.append(f'<td style="text-align: center;">{value}</td>')
        rows.append(
            f'<tr><th scope="row">{html.escape(model_name)}</th>{"".join(cells)}</tr>'
        )

    if not rows:
        return ""

    return f"""
        <div style="margin-top: 1.5rem;">
            <h3 style="margin-bottom: 0.25rem;">Validated Model/GPU Configurations</h3>
            <small style="color: #666;">Maximum validated concurrent tasks. A dash means this model/GPU combination has not been validated.</small>
            <div style="overflow-x: auto; margin-top: 0.5rem;">
                <table style="min-width: max-content; margin-bottom: 0;">
                    <thead><tr><th scope="col">Model</th>{header_cells}</tr></thead>
                    <tbody>{"".join(rows)}</tbody>
                </table>
            </div>
        </div>
    """
