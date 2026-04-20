import argparse
import csv
import json
import os
import re
from typing import Dict, List, Optional, Tuple


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_WS_ROOT = os.path.join(PROJECT_ROOT, "ws", "chapter3_runs")
DEFAULT_OUTPUT_CSV = os.path.join(PROJECT_ROOT, "reports", "long_results.csv")

EXPERIMENT_MODEL_SUFFIX = {
    "full_model": "",
    "wo_attention": " (w/o attention)",
    "wo_query": " (w/o query)",
    "wo_weight": " (w/o weight)",
    "wo_consistency": " (w/o consistency)",
}

DATASET_NAME_MAP = {
    "assist2009": "ASSIST2009",
    "assist_2009": "ASSIST2009",
    "assist09": "ASSIST2009",
    "assist2012": "ASSIST2012",
    "assist_2012": "ASSIST2012",
    "kddcup": "KDDCup",
    "kddcup2010": "KDDCup",
    "kdd_cup": "KDDCup",
}


def read_json(path: str) -> Dict:
    with open(path, "r", encoding="utf8") as f:
        return json.load(f)


def read_jsonl(path: str) -> List[Dict]:
    records: List[Dict] = []
    with open(path, "r", encoding="utf8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def latest_final_summary(records: List[Dict]) -> Optional[Dict]:
    summaries = [record for record in records if record.get("dtype") == "final_summary"]
    if not summaries:
        return None
    return summaries[-1]


def normalize_dataset_name(dataset_name: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "", dataset_name.strip().lower())
    return DATASET_NAME_MAP.get(normalized, dataset_name.upper())


def parse_attr_idx(workspace_name: str, config: Dict) -> Optional[int]:
    attr_idx = config.get("attr_idx")
    if attr_idx is not None:
        return int(attr_idx)
    match = re.search(r"DIRT_(\d+)$", workspace_name)
    if not match:
        return None
    return int(match.group(1))


def normalize_model_name(experiment_name: str, workspace_name: str, config: Dict) -> str:
    attr_idx = parse_attr_idx(workspace_name, config)
    if experiment_name in EXPERIMENT_MODEL_SUFFIX and attr_idx is not None:
        base_name = f"DIRT+_{attr_idx}"
        suffix = EXPERIMENT_MODEL_SUFFIX[experiment_name]
        return f"{base_name}{suffix}"
    return workspace_name


def infer_experiment_name(ws_root: str, workspace_dir: str) -> Optional[str]:
    relative_parts = os.path.relpath(workspace_dir, ws_root).split(os.sep)
    if len(relative_parts) < 2:
        return None
    return relative_parts[-2]


def iter_workspace_dirs(ws_root: str):
    for root, dirs, files in os.walk(ws_root):
        if "metrics.jsonl" in files and "model_config.txt" in files:
            yield root
            dirs[:] = []


def collect_long_rows(ws_root: str) -> Tuple[List[Dict], List[str]]:
    rows: List[Dict] = []
    cleanup_notes: List[str] = []
    seen: Dict[Tuple[str, str, int], Dict] = {}

    if not os.path.exists(ws_root):
        return rows, cleanup_notes

    for ws_dir in sorted(iter_workspace_dirs(ws_root)):
        metrics_path = os.path.join(ws_dir, "metrics.jsonl")
        config_path = os.path.join(ws_dir, "model_config.txt")

        records = read_jsonl(metrics_path)
        final_summary = latest_final_summary(records)
        if final_summary is None:
            cleanup_notes.append(f"Skipped incomplete workspace without final_summary: {ws_dir}")
            continue

        config = read_json(config_path)
        dataset_name = str(config.get("data", "")).strip()
        seed_value = config.get("seed")
        if not dataset_name:
            cleanup_notes.append(f"Skipped workspace without dataset name: {ws_dir}")
            continue
        if seed_value is None:
            cleanup_notes.append(f"Skipped workspace without seed: {ws_dir}")
            continue

        experiment_name = infer_experiment_name(ws_root, ws_dir) or str(config.get("exp_name", "")).strip()
        workspace_name = os.path.basename(ws_dir)
        dataset = normalize_dataset_name(dataset_name)
        seed = int(seed_value)
        model = normalize_model_name(experiment_name, workspace_name, config)
        row = {
            "模型": model,
            "数据集": dataset,
            "种子": seed,
            "AUC": float(final_summary.get("test_auc", 0.0)),
            "ACC": float(final_summary.get("test_acc", 0.0)),
        }

        dedupe_key = (row["模型"], row["数据集"], row["种子"])
        existing = seen.get(dedupe_key)
        if existing is None:
            seen[dedupe_key] = row
            continue

        if existing["AUC"] == row["AUC"] and existing["ACC"] == row["ACC"]:
            cleanup_notes.append(f"Removed exact duplicate row for key={dedupe_key}")
            continue

        raise ValueError(
            f"Conflicting duplicate results found for key={dedupe_key}: "
            f"existing={existing}, incoming={row}"
        )

    rows = sorted(
        seen.values(),
        key=lambda item: (item["数据集"], item["模型"], item["种子"]),
    )
    return rows, cleanup_notes


def write_csv(path: str, rows: List[Dict]) -> None:
    fieldnames = ["模型", "数据集", "种子", "AUC", "ACC"]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(description="Export experiment results as a long-format CSV.")
    parser.add_argument("--ws_root", type=str, default=DEFAULT_WS_ROOT)
    parser.add_argument("--output_csv", type=str, default=DEFAULT_OUTPUT_CSV)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    rows, cleanup_notes = collect_long_rows(args.ws_root)
    write_csv(args.output_csv, rows)
    print(f"[done] rows={len(rows)}, output={args.output_csv}")
    if cleanup_notes:
        print("[cleanup]")
        for note in cleanup_notes:
            print(note)
