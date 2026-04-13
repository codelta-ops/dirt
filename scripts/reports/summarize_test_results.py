import argparse
import csv
import json
import os
import re
from typing import Dict, List, Optional, Tuple


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def read_jsonl(path: str) -> List[Dict]:
    records: List[Dict] = []
    with open(path, "r", encoding="utf8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"[warn] skip malformed json line: {path}:{line_no}")
    return records


def read_json(path: str) -> Dict:
    with open(path, "r", encoding="utf8") as f:
        return json.load(f)


def validation_priority_key(record: Dict) -> Tuple[float, float, float, int]:
    return (
        float(record.get("auc", 0.0)),
        float(record.get("acc", 0.0)),
        -float(record.get("rmse", float("inf"))),
        -int(record.get("epoch", 0)),
    )


def find_best_validation_record(records: List[Dict], stage: int) -> Optional[Dict]:
    candidates = [
        r for r in records if r.get("dtype") == "validation" and int(r.get("stage", -1)) == int(stage)
    ]
    if not candidates:
        return None
    return max(candidates, key=validation_priority_key)


def parse_attr_idx_from_name(name: str) -> Optional[int]:
    match = re.search(r"DIRT_(\d+)$", name)
    if not match:
        return None
    return int(match.group(1))


def find_workspace_dirs(ws_root: str) -> List[str]:
    found = []
    for root, _, files in os.walk(ws_root):
        if "metrics.jsonl" in files:
            found.append(root)
    return sorted(found)


def pick_last_test_record(records: List[Dict]) -> Optional[Dict]:
    tests = [r for r in records if r.get("dtype") == "test"]
    if not tests:
        return None
    return tests[-1]


def collect_workspace_row(ws_dir: str) -> Optional[Dict]:
    metrics_path = os.path.join(ws_dir, "metrics.jsonl")
    config_path = os.path.join(ws_dir, "model_config.txt")
    if not os.path.exists(metrics_path):
        return None

    records = read_jsonl(metrics_path)
    test_record = pick_last_test_record(records)
    if test_record is None:
        print(f"[warn] no test record: {ws_dir}")
        return None

    config = {}
    if os.path.exists(config_path):
        try:
            config = read_json(config_path)
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] failed to read config {config_path}: {exc}")

    stage = int(test_record.get("stage", -1))
    best_val = find_best_validation_record(records, stage=stage)

    ws_name = os.path.basename(ws_dir.rstrip("/\\"))
    attr_idx = config.get("attr_idx")
    if attr_idx is None:
        attr_idx = parse_attr_idx_from_name(ws_name)

    row = {
        "workspace": ws_name,
        "workspace_path": ws_dir,
        "attr_idx": attr_idx if attr_idx is not None else "",
        "test_stage": stage,
        "test_epoch": int(test_record.get("epoch", -1)),
        "test_auc": float(test_record.get("auc", 0.0)),
        "test_acc": float(test_record.get("acc", 0.0)),
        "test_rmse": float(test_record.get("rmse", 0.0)),
        "test_pred_loss": float(test_record.get("pred_loss", 0.0)),
        "best_val_epoch_same_stage": int(best_val.get("epoch", -1)) if best_val else "",
        "best_val_auc_same_stage": float(best_val.get("auc", 0.0)) if best_val else "",
        "best_val_acc_same_stage": float(best_val.get("acc", 0.0)) if best_val else "",
        "best_val_rmse_same_stage": float(best_val.get("rmse", 0.0)) if best_val else "",
        "use_multihead_temporal_attn": config.get("use_multihead_temporal_attn", ""),
        "use_multihead_query_attn": config.get("use_multihead_query_attn", ""),
        "multihead_temporal_num_heads": config.get("multihead_temporal_num_heads", ""),
        "multihead_query_num_heads": config.get("multihead_query_num_heads", ""),
        "use_confidence_consistency": config.get("use_confidence_consistency", ""),
        "confidence_consistency_mode": config.get("confidence_consistency_mode", ""),
        "seed": config.get("seed", ""),
    }
    return row


def sort_rows(rows: List[Dict]) -> List[Dict]:
    def key(row: Dict) -> Tuple[int, str]:
        attr = row.get("attr_idx")
        if attr == "":
            return (10**9, row["workspace"])
        return (int(attr), row["workspace"])

    return sorted(rows, key=key)


def print_console_summary(rows: List[Dict]) -> None:
    if not rows:
        print("No rows to summarize.")
        return
    print("")
    print("Test Summary")
    print("-" * 100)
    header = (
        f"{'workspace':<18} {'attr':<4} {'stage':<5} {'epoch':<5} "
        f"{'AUC':<10} {'ACC':<10} {'RMSE':<10} {'val_epoch':<8}"
    )
    print(header)
    print("-" * 100)
    for r in rows:
        print(
            f"{str(r['workspace']):<18} "
            f"{str(r['attr_idx']):<4} "
            f"{str(r['test_stage']):<5} "
            f"{str(r['test_epoch']):<5} "
            f"{r['test_auc']:<10.6f} "
            f"{r['test_acc']:<10.6f} "
            f"{r['test_rmse']:<10.6f} "
            f"{str(r['best_val_epoch_same_stage']):<8}"
        )
    print("-" * 100)
    avg_auc = sum(float(r["test_auc"]) for r in rows) / len(rows)
    avg_acc = sum(float(r["test_acc"]) for r in rows) / len(rows)
    avg_rmse = sum(float(r["test_rmse"]) for r in rows) / len(rows)
    print(f"avg_test_auc={avg_auc:.6f}, avg_test_acc={avg_acc:.6f}, avg_test_rmse={avg_rmse:.6f}")
    print("")


def write_csv(rows: List[Dict], output_csv: str) -> None:
    if not rows:
        with open(output_csv, "w", encoding="utf8", newline="") as f:
            f.write("")
        return
    fieldnames = list(rows[0].keys())
    with open(output_csv, "w", encoding="utf8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect and summarize DIRT/DIRT+ test metrics.")
    parser.add_argument("--ws_root", type=str, default="ws/dirt_plus/assist09", help="Workspace root directory.")
    parser.add_argument("--output_csv", type=str, default="", help="Output CSV path. Default: <ws_root>/test_summary.csv")
    args = parser.parse_args()
    os.chdir(PROJECT_ROOT)

    ws_root = args.ws_root
    if not os.path.exists(ws_root):
        raise FileNotFoundError(f"Workspace root does not exist: {ws_root}")

    output_csv = args.output_csv or os.path.join(ws_root, "test_summary.csv")
    ws_dirs = find_workspace_dirs(ws_root)
    if not ws_dirs:
        print(f"[warn] no metrics.jsonl found under: {ws_root}")
        write_csv([], output_csv)
        print(f"[done] wrote empty file: {output_csv}")
        return

    rows: List[Dict] = []
    for ws_dir in ws_dirs:
        row = collect_workspace_row(ws_dir)
        if row is not None:
            rows.append(row)
    rows = sort_rows(rows)

    write_csv(rows, output_csv)
    print_console_summary(rows)
    print(f"[done] csv saved: {output_csv}")
    print(f"[done] workspace_count={len(ws_dirs)}, summarized_count={len(rows)}")


if __name__ == "__main__":
    main()
