"""
Builds each generated task's environment/mounts.json manifest: absolute-path
bind mount specs for the six scenario input subpaths, resolved at generation
time (see `_common.INPUT_SUBPATHS`).
"""

from __future__ import annotations

from pathlib import Path

from ._common import INPUT_SUBPATHS
from .scenarios import _resolve_alerts_mount


def build_mounts_manifest(scenario_dir: Path) -> list[dict]:
    """Written at generation time (when the scenario dir is known) so that
    `harbor run` and our custom PodmanEnvironment can apply them without
    requiring DATA_ROOT at run time. Paths are machine-specific; tasks must
    be regenerated if data moves (datasets/ is gitignored). "source_rel" is
    kept alongside "source" for reference.
    """
    mounts = []
    for sub in INPUT_SUBPATHS:
        if sub == "alerts":
            source_rel, target_rel = _resolve_alerts_mount(scenario_dir)
        else:
            source_rel = target_rel = sub
        mounts.append((source_rel, target_rel))

    manifest = []
    for source_rel, target_rel in mounts:
        source_path = scenario_dir / source_rel
        target = f"/workspace/{target_rel}"
        if source_path.is_dir():
            # Mount the directory itself so the target path is created in
            # the container. HF Hub cache uses relative symlinks inside
            # snapshot dirs (e.g. foo.json -> ../../../../../blobs/HASH);
            # those break inside the container because the blobs tree
            # isn't at that path. Fix: for each symlinked file inside the
            # directory, add an extra mount of its resolved real path at
            # the exact target file path -- this shadows the broken
            # symlink with the actual content without needing a blobs dir.
            manifest.append({
                "type": "bind",
                "source": str(source_path),
                "source_rel": source_rel,
                "target": target,
                "read_only": True,
            })
            for child in sorted(source_path.iterdir()):
                real_child = child.resolve()
                if real_child != child and real_child.is_file():
                    manifest.append({
                        "type": "bind",
                        "source": str(real_child),
                        "target": f"{target}/{child.name}",
                        "read_only": True,
                    })
        else:
            # Individual file: resolve any symlink on the host so Podman
            # bind-mounts the actual blob content rather than a symlink
            # whose target may not be accessible inside the container.
            manifest.append({
                "type": "bind",
                "source": str(source_path.resolve()),
                "source_rel": source_rel,
                "target": target,
                "read_only": True,
            })
    return manifest
