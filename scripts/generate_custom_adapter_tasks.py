#!/usr/bin/env python3
"""Generate Harbor task(s) for a custom adapter, optionally verifying them
with Harbor's (non-LLM) oracle agent, which replays each task's baked-in
solve.sh to prove the task is solvable and gradable -- not that a model
can solve it.

Generation must run from adapters/<adapter-id> (so the adapter's own module
resolves); verification must run from the repo root (so the relative
datasets/<adapter-id>/harbor_tasks path resolves). That path is a
repo-local convention -- revert before upstreaming.

Usage: generate_custom_adapter_tasks.py <adapter-id> [<task-id>] [--verify]
       [--podman] [--host-network]
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

_COLOR = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None
_BOLD, _DIM, _CYAN, _GREEN, _RESET = "\033[1m", "\033[2m", "\033[36m", "\033[32m", "\033[0m"


def style(text: str, *codes: str) -> str:
    """Wrap text in ANSI codes, unless stdout isn't a TTY or NO_COLOR is set."""
    return f"{''.join(codes)}{text}{_RESET}" if _COLOR else text


def repo_root() -> Path:
    """Resolve the repo root via git, so this works from any caller cwd.

    Assumes adapters/ dirs aren't themselves separate git repos; true today.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"error: not inside a git repo: {e}", file=sys.stderr)
        sys.exit(2)
    return Path(result.stdout.strip())


def parse_args() -> argparse.Namespace:
    """Parse: <adapter-id> [<task-id>] [--verify]."""
    parser = argparse.ArgumentParser(
        description="Generate (and optionally verify) Harbor task(s) for a custom adapter.",
    )
    parser.add_argument("adapter_id", help="Adapter id, i.e. adapters/<adapter-id>.")
    parser.add_argument(
        "task_id", nargs="?", default=None,
        help="Optional: act on just this one task. Omit for every task in the dataset.",
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="After generating, run Harbor's oracle and require reward = 1.",
    )
    parser.add_argument(
        "--podman", action="store_true",
        help="Force verifying against Podman instead of Docker (adds "
             "--environment-import-path coding_agent_bench.helpers.podman:"
             "PodmanEnvironment to the oracle run). Usually not needed -- Docker is "
             "used automatically when it's available/running, with an automatic "
             "fallback to Podman otherwise. No effect without --verify.",
    )
    parser.add_argument(
        "--host-network", action="store_true",
        help="Give the container host networking (to reach a host-local model server, "
             "e.g. Ollama). With --podman: sets PODMAN_HOST_NETWORK=1 for the oracle "
             "subprocess. Without --podman (real Docker): appends --extra-docker-compose "
             "pointing at the task's environment/host-network-overlay.yaml.example, if the "
             "generated task has one. No effect without --verify.",
    )
    return parser.parse_args()


def check_requirements(root: Path, adapter_dir: Path, need_harbor: bool) -> None:
    """Fail fast with a clear error if uv, the adapter, or harbor aren't available."""
    if shutil.which("uv") is None:
        print("error: 'uv' not found on PATH", file=sys.stderr)
        sys.exit(2)

    if not adapter_dir.is_dir():
        print(f"error: adapter not found: {adapter_dir}", file=sys.stderr)
        adapters_dir = adapter_dir.parent
        available = (
            sorted(p.name for p in adapters_dir.iterdir() if p.is_dir())
            if adapters_dir.is_dir() else []
        )
        if available:
            print(f"available adapters: {', '.join(available)}", file=sys.stderr)
        sys.exit(2)

    if need_harbor:
        result = subprocess.run(
            ["uv", "run", "harbor", "--help"], cwd=root, capture_output=True, text=True,
        )
        if result.returncode != 0:
            print(
                "error: 'uv run harbor' failed from the repo root -- is harbor installed? (uv sync)",
                file=sys.stderr,
            )
            sys.exit(2)


def _engine_available(binary: str) -> bool:
    """True if `binary` is on PATH and `<binary> info` actually succeeds."""
    if shutil.which(binary) is None:
        return False
    try:
        result = subprocess.run(
            [binary, "info"], capture_output=True, timeout=10,
        )
    except (subprocess.TimeoutExpired, OSError):
        return False
    return result.returncode == 0


def _is_real_docker() -> bool:
    """True only for an actual Docker Engine -- not a `docker` shim/alias to podman.

    A `docker` binary that's really podman underneath (e.g. `alias docker=podman`,
    or a wrapper script) makes `docker info` succeed even though `docker compose`
    still breaks (see this script's --podman flag). `docker version`'s plain
    (non-JSON) output says "Client: Podman Engine" in that case, vs.
    "Client: Docker Engine" for the real thing -- check for that literal string
    rather than trusting `docker info`'s exit code alone.
    """
    if not _engine_available("docker"):
        return False
    try:
        result = subprocess.run(
            ["docker", "version"], capture_output=True, text=True, timeout=10,
        )
    except (subprocess.TimeoutExpired, OSError):
        return False
    return "podman" not in result.stdout.lower()


def resolve_use_podman(explicit_podman: bool) -> bool:
    """Decide whether the oracle run should use Podman.

    Respects an explicit --podman as-is (no detection). Otherwise prefers
    Docker if it's actually a real Docker Engine; falls back to Podman
    otherwise (including when `docker` is really just a shim for podman),
    since that's this repo's only other supported engine. Exits with a
    clear error if neither is usable.
    """
    if explicit_podman:
        return True

    if _is_real_docker():
        return False

    print(style("› no real Docker Engine available -- falling back to Podman", _BOLD, _CYAN), flush=True)
    if _engine_available("podman"):
        return True

    print(
        "error: neither Docker nor Podman is available/running on this machine. "
        "Install/start one of them, or pass --podman explicitly if Podman just "
        "needs a machine started (e.g. `podman machine start`).",
        file=sys.stderr,
    )
    sys.exit(2)


def run_cmd(cmd: list[str], cwd: Path, root: Path, extra_env: dict[str, str] | None = None) -> None:
    """Run cmd from cwd, streaming output live; exit early on non-zero return."""
    print(style(f"$ {' '.join(cmd)}", _DIM), flush=True)
    if cwd != root:
        print(style(f"  (in {cwd.relative_to(root)})", _DIM), flush=True)
    # uv warns when VIRTUAL_ENV (set by an activated parent-repo .venv) doesn't
    # match the adapter's own project env -- noise, since we always want the
    # adapter's own env here, not whatever happens to be active.
    env = os.environ.copy()
    env.pop("VIRTUAL_ENV", None)
    if extra_env:
        env.update(extra_env)
    result = subprocess.run(cmd, cwd=cwd, env=env)
    if result.returncode != 0:
        sys.exit(result.returncode)


def _build_mounts_flag(verify_path: Path, task_id: str) -> list[str]:
    """Build ["--mounts", <json>] from environment/mounts.json, if the task has one.

    Returns [] silently if there's no mounts.json (e.g. itbench-lite tasks,
    which never have one). Reads the absolute "source" field written at
    generation time -- no DATA_ROOT env var required.
    """
    mounts_path = verify_path / "environment" / "mounts.json"
    if not mounts_path.is_file():
        return []

    manifest = json.loads(mounts_path.read_text())

    # Prefer the absolute "source" field written at generation time.
    # Fall back to resolving "source_rel" via DATA_ROOT for legacy mounts.json.
    mounts = []
    for entry in manifest:
        if "source" in entry:
            source = entry["source"]
        else:
            data_root = os.environ.get("DATA_ROOT")
            if not data_root:
                print(
                    f"error: {mounts_path} has no absolute 'source' field and "
                    "DATA_ROOT is not set. Regenerate tasks or set DATA_ROOT.",
                    file=sys.stderr,
                )
                sys.exit(2)
            candidates = [task_id]
            match = re.search(r"(\d+)$", task_id)
            if match:
                candidates.append(f"Scenario-{match.group(1)}")
            tried = [Path(data_root) / c for c in candidates]
            scenario_dir = next((p for p in tried if p.is_dir()), None)
            if scenario_dir is None:
                print(
                    "error: could not find scenario directory under DATA_ROOT; tried:\n"
                    + "\n".join(f"  {p}" for p in tried),
                    file=sys.stderr,
                )
                sys.exit(2)
            source = str(scenario_dir / entry["source_rel"])
        mounts.append({
            "type": entry["type"],
            "source": source,
            "target": entry["target"],
            **({"read_only": True} if entry.get("read_only") else {}),
        })
    return ["--mounts", json.dumps(mounts)]


def _build_host_network_args(verify_path: Path, podman: bool) -> tuple[list[str], dict[str, str] | None]:
    """Return (extra harbor_cmd args, extra_env) that grant host networking."""
    if podman:
        return [], {"PODMAN_HOST_NETWORK": "1"}

    overlay_path = verify_path / "environment" / "host-network-overlay.yaml.example"
    if not overlay_path.is_file():
        print(
            f"error: --host-network without --podman requires {overlay_path} to exist "
            "(there's no other way to enable host networking for real Docker here).",
            file=sys.stderr,
        )
        sys.exit(2)
    return ["--extra-docker-compose", str(overlay_path)], None


def main() -> None:
    """Always generate; run the oracle too if --verify was passed."""
    args = parse_args()
    root = repo_root()
    adapter_dir = root / "adapters" / args.adapter_id
    check_requirements(root, adapter_dir, need_harbor=args.verify)

    tasks_dir = root / "datasets" / args.adapter_id / "harbor_tasks"
    # adapter id "itbench-lite" -> code dir "itbench_lite" (dashes to underscores).
    adapter_code_dir = args.adapter_id.replace("-", "_")
    rel_tasks_dir = os.path.relpath(tasks_dir, adapter_dir)

    gen_cmd = [
        "uv", "run", "python", "-m", f"{adapter_code_dir}.main",
        "--output-dir", rel_tasks_dir,
        "--overwrite",
    ]
    if args.task_id:
        gen_cmd += ["--task-ids", args.task_id]  # accepts nargs="+", one id is fine

    suffix = f" (task-id: {args.task_id})" if args.task_id else ""
    print(style(f"› [{args.adapter_id}] generating task(s){suffix}", _BOLD, _CYAN), flush=True)
    run_cmd(gen_cmd, cwd=adapter_dir, root=root)

    # This adapter folder will eventually move upstream to Harbor's own repo,
    # where this script won't exist -- so these pointers live here, not baked
    # into any adapter-generated file, to avoid going stale after that move.
    print(style(f"\n✓ tasks written to {tasks_dir.relative_to(root)}", _GREEN))
    print(f"  see {(adapter_dir / 'README.md').relative_to(root)} for usage, flags, and troubleshooting", flush=True)

    if not args.verify:
        print(style("\n✓ done (generate-only).", _GREEN) + " Re-run with --verify to check it against the Harbor oracle.")
        return

    verify_path = tasks_dir / args.task_id if args.task_id else tasks_dir
    rel_verify_path = verify_path.relative_to(root)

    use_podman = resolve_use_podman(args.podman)

    harbor_cmd = ["uv", "run", "harbor", "run", "-p", str(rel_verify_path), "-a", "oracle"]
    extra_env: dict[str, str] | None = None

    if use_podman:
        harbor_cmd += [
            "--env",
            "coding_agent_bench.helpers.podman:PodmanEnvironment",
        ]

    # --mounts only makes sense for a single task -- one JSON blob can't
    # sensibly cover every task's differing scenario data in a whole dir.
    if args.task_id:
        harbor_cmd += _build_mounts_flag(verify_path, args.task_id)

    if args.host_network:
        extra_args, extra_env = _build_host_network_args(verify_path, use_podman)
        harbor_cmd += extra_args

    print(style(f"\n› running oracle on {rel_verify_path} (must reach reward = 1)", _BOLD, _CYAN), flush=True)
    run_cmd(harbor_cmd, cwd=root, root=root, extra_env=extra_env)


if __name__ == "__main__":
    main()
