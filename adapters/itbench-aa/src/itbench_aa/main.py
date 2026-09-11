"""
Main entry point for the template adapter. Do not modify any of the existing flags.
You can add any additional flags you need.

Constructs the Adapter class defined in adapter.py and calls run() to generate tasks
in the Harbor format at the configured output directory.
"""

import argparse
from pathlib import Path

from .adapter import ITBenchAAAdapter

# Default output dir: <repo>/datasets/<adapter_id>
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[4] / "datasets" / "itbench-aa"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write generated tasks",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Generate only the first N tasks",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing tasks",
    )
    parser.add_argument(
        "--task-ids",
        nargs="+",
        default=None,
        help="Only generate these task IDs",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Override the auto-detected dataset root (use if your HF download landed "
             "in a different location or layout than datasets/itbench-aa/sre/)",
    )
    args = parser.parse_args()

    adapter = ITBenchAAAdapter(
        args.output_dir,
        overwrite=args.overwrite,
        limit=args.limit,
        task_ids=args.task_ids,
        data_root=args.data_root,
    )

    adapter.run()


if __name__ == "__main__":
    main()
