import os
import re
import shlex
from pathlib import Path

from harbor.skills import resolve_repo_source

_ENV_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
DEFAULT_STORAGE_ENDPOINT_URL = "http://harbor-storage:9000"


def storage_endpoint_url() -> str:
    """Return the configured storage API endpoint, with an in-cluster default."""
    configured = os.environ.get("STORAGE_ENDPOINT_URL", "").strip().rstrip("/")
    return configured or DEFAULT_STORAGE_ENDPOINT_URL


def cmd_to_string(cmd: list[str]):
    """Format a bash command as a string."""
    cmd_string = shlex.join(cmd)
    return cmd_string


def validate_remote_skill_sources(skills: list[str] | None) -> None:
    """Require remotely executed skills to use Harbor-supported Git sources."""
    for skill in skills or []:
        try:
            if skill.startswith((".", "/", "~")) or Path(skill).exists():
                raise ValueError
            resolve_repo_source(skill)
        except ValueError as exc:
            raise ValueError(
                f"Remote skill source {skill!r} is not a Git source. "
                "Use org/name[@ref] or an HTTP(S) Git URL; local paths are "
                "only supported for locally orchestrated runs."
            ) from exc


def is_stage_environment() -> bool:
    """Return True if the application is running in stage environment."""
    environment = os.environ.get("ENVIRONMENT", "").lower()
    if environment not in {"prod", "stage"}:
        raise ValueError("ENVIRONMENT must be set to either 'prod' or 'stage'")
    return environment == "stage"


def parse_envs(envs: str | list[str] | None) -> dict[str, str]:
    """Parse repeatable `key=value` entries into a dict.

    Each entry may itself be a comma-separated `key=value,key=value` string,
    preserving the original single-flag form. Later entries override earlier
    ones on key collision.
    """
    if not envs:
        return {}
    entries = [envs] if isinstance(envs, str) else envs
    parsed = {}
    for entry in entries:
        for pair in entry.split(","):
            pair = pair.strip()
            if not pair:
                continue
            if "=" not in pair:
                raise ValueError(f"Invalid --envs entry (expected key=value): {pair!r}")
            key, value = pair.split("=", 1)
            key = key.strip()
            if not _ENV_KEY_RE.match(key):
                raise ValueError(
                    f"Invalid --envs entry (variable name must match [A-Za-z_][A-Za-z0-9_]*): {pair!r}"
                )
            parsed[key] = value
    return parsed


def validate_key_value_entries(
    entries: list[str] | None, flag: str, *, env_keys: bool = False
) -> None:
    """Validate repeatable KEY=VALUE passthrough entries for a CLI flag.

    Values are passed to Harbor verbatim; only the key side is checked. Empty
    values (`KEY=`) are legal. With `env_keys=True` the key must also be a valid
    environment variable name.
    """
    for entry in entries or []:
        if "=" not in entry:
            raise ValueError(f"Invalid {flag} entry (expected key=value): {entry!r}")
        key = entry.split("=", 1)[0].strip()
        if not key:
            raise ValueError(f"Invalid {flag} entry (empty key): {entry!r}")
        if env_keys and not _ENV_KEY_RE.match(key):
            raise ValueError(
                f"Invalid {flag} entry (variable name must match [A-Za-z_][A-Za-z0-9_]*): {entry!r}"
            )


def envs_to_export_lines(envs: dict[str, str]) -> str:
    """Format env vars as `export KEY=VALUE` lines, for display purposes only."""
    return "\n".join(f"export {key}={shlex.quote(value)}" for key, value in envs.items())
