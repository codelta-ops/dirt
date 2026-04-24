import argparse
import os
import shutil
from typing import List


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_TARGETS = [
    os.path.join(PROJECT_ROOT, "ws", "chapter3_runs"),
    os.path.join(PROJECT_ROOT, "ws", "dirt"),
]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def reset_targets(targets: List[str], dry_run: bool = False) -> None:
    for target in targets:
        normalized = os.path.abspath(target)
        if not normalized.startswith(os.path.join(PROJECT_ROOT, "ws")):
            raise ValueError(f"Refusing to delete path outside workspace outputs: {normalized}")

        if not os.path.exists(normalized):
            print(f"[skip] missing: {normalized}")
            ensure_dir(normalized)
            continue

        print(f"[delete] {normalized}")
        if not dry_run:
            shutil.rmtree(normalized)
            ensure_dir(normalized)


def parse_args():
    parser = argparse.ArgumentParser(description="Delete local experiment outputs under ws/ and recreate empty roots.")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--targets", nargs="*", default=DEFAULT_TARGETS)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    reset_targets(args.targets, dry_run=args.dry_run)
