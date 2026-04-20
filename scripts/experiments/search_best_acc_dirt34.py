import argparse
import json
import os
import subprocess
from typing import Dict, List, Optional


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_MODEL_SCRIPT = os.path.join("models", "dirt_plus.py")


EXPERIMENTS: List[Dict] = [
    {
        "name": "acc_plain_control",
        "purpose": "Current plain-loss control setting on DIRT_3/4.",
        "args": {
            "loss_weight_mode": "plain",
            "stage1_lr": 0.002,
            "stage2_lr": 0.0006,
            "lambda_consistency": 0.2,
            "consistency_warmup_epochs": 4,
        },
    },
    {
        "name": "acc_plain_s2lr0006",
        "purpose": "Lower stage2 lr to maximize validation ACC on DIRT_3/4.",
        "args": {
            "loss_weight_mode": "plain",
            "stage1_lr": 0.002,
            "stage2_lr": 0.0006,
            "lambda_consistency": 0.2,
            "consistency_warmup_epochs": 4,
        },
    },
    {
        "name": "acc_plain_s2lr0007",
        "purpose": "Moderately lower stage2 lr for a softer stage2 update.",
        "args": {
            "loss_weight_mode": "plain",
            "stage1_lr": 0.002,
            "stage2_lr": 0.0007,
            "lambda_consistency": 0.2,
            "consistency_warmup_epochs": 4,
        },
    },
    {
        "name": "acc_plain_lambda015",
        "purpose": "Slightly lower consistency weight and check ACC sensitivity.",
        "args": {
            "loss_weight_mode": "plain",
            "stage1_lr": 0.002,
            "stage2_lr": 0.0006,
            "lambda_consistency": 0.15,
            "consistency_warmup_epochs": 4,
        },
    },
    {
        "name": "acc_plain_warmup5",
        "purpose": "Longer consistency warmup for a gentler stage2 transition.",
        "args": {
            "loss_weight_mode": "plain",
            "stage1_lr": 0.002,
            "stage2_lr": 0.0006,
            "lambda_consistency": 0.2,
            "consistency_warmup_epochs": 5,
        },
    },
]


def format_value(value):
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def build_command(args, exp_config):
    cmd = [
        args.python_bin,
        args.script,
        "--data", args.data,
        "--cross_idx", str(args.cross_idx),
        "--device", args.device,
        "--stage1_epochs", str(args.stage1_epochs),
        "--stage2_epochs", str(args.stage2_epochs),
        "--ws_root", args.ws_root,
        "--exp_name", exp_config["name"],
        "--attr_indices", args.attr_indices,
        "--seed", str(args.seed),
    ]
    for key, value in exp_config["args"].items():
        cmd.extend([f"--{key}", format_value(value)])
    return cmd


def build_label(exp_config):
    pieces = [f"{k}={format_value(v)}" for k, v in exp_config["args"].items()]
    return f'{exp_config["name"]} | ' + ", ".join(pieces)


def print_commands(args):
    for exp in EXPERIMENTS:
        print(f"# Experiment: {build_label(exp)}")
        print(f"# Purpose: {exp['purpose']}")
        print(" ".join(build_command(args, exp)))
        print()


def run_commands(args):
    for idx, exp in enumerate(EXPERIMENTS, start=1):
        cmd = build_command(args, exp)
        print(f">>> Running [{idx}/{len(EXPERIMENTS)}] {build_label(exp)}")
        print(f">>> Purpose: {exp['purpose']}")
        print(" ".join(cmd))
        subprocess.run(cmd, check=True)


def read_summary_lines(summary_path: str) -> List[str]:
    if not os.path.exists(summary_path):
        return []
    with open(summary_path, "r", encoding="utf8") as f:
        return [line.strip() for line in f if line.strip()]


def parse_metric(line: str, key: str) -> Optional[float]:
    marker = f"{key}="
    if marker not in line:
        return None
    tail = line.split(marker, 1)[1]
    value = tail.split(",", 1)[0].strip()
    try:
        return float(value)
    except ValueError:
        return None


def parse_attr(line: str) -> Optional[str]:
    if not line.startswith("attr="):
        return None
    return line.split(",", 1)[0].split("=", 1)[1].strip()


def summary_rows(args) -> List[Dict]:
    rows: List[Dict] = []
    for exp in EXPERIMENTS:
        summary_path = os.path.join(args.ws_root, exp["name"], "experiment_summary.txt")
        lines = read_summary_lines(summary_path)
        for line in lines:
            attr = parse_attr(line)
            if attr not in {"DIRT_3", "DIRT_4"}:
                continue
            rows.append(
                {
                    "experiment": exp["name"],
                    "attr": attr,
                    "validation_auc": parse_metric(line, "validation_auc"),
                    "validation_acc": parse_metric(line, "validation_acc"),
                    "validation_rmse": parse_metric(line, "validation_rmse"),
                    "test_auc": parse_metric(line, "test_auc"),
                    "test_acc": parse_metric(line, "test_acc"),
                    "test_rmse": parse_metric(line, "test_rmse"),
                    "best_epoch": parse_metric(line, "best_epoch"),
                }
            )
    return rows


def print_summaries(args):
    rows = summary_rows(args)
    if not rows:
        print("No summary rows found.")
        return

    rows.sort(
        key=lambda r: (
            float("-inf") if r["validation_acc"] is None else r["validation_acc"],
            float("-inf") if r["validation_auc"] is None else r["validation_auc"],
            float("inf") if r["validation_rmse"] is None else -r["validation_rmse"],
        ),
        reverse=True,
    )

    print("===== Ranked By Validation ACC (DIRT_3/4 only) =====")
    for row in rows:
        print(
            f'{row["experiment"]} | {row["attr"]} | '
            f'val_acc={format_value(row["validation_acc"])} | '
            f'val_auc={format_value(row["validation_auc"])} | '
            f'val_rmse={format_value(row["validation_rmse"])} | '
            f'test_acc={format_value(row["test_acc"])} | '
            f'test_auc={format_value(row["test_auc"])} | '
            f'test_rmse={format_value(row["test_rmse"])} | '
            f'best_epoch={format_value(row["best_epoch"])}'
        )

    per_exp: Dict[str, Dict[str, float]] = {}
    for exp in EXPERIMENTS:
        exp_name = exp["name"]
        exp_rows = [r for r in rows if r["experiment"] == exp_name]
        if not exp_rows:
            continue
        valid_accs = [r["validation_acc"] for r in exp_rows if r["validation_acc"] is not None]
        valid_aucs = [r["validation_auc"] for r in exp_rows if r["validation_auc"] is not None]
        valid_rmses = [r["validation_rmse"] for r in exp_rows if r["validation_rmse"] is not None]
        if not valid_accs:
            continue
        per_exp[exp_name] = {
            "mean_validation_acc": sum(valid_accs) / len(valid_accs),
            "mean_validation_auc": sum(valid_aucs) / len(valid_aucs) if valid_aucs else None,
            "mean_validation_rmse": sum(valid_rmses) / len(valid_rmses) if valid_rmses else None,
        }

    if per_exp:
        ranked_exp = sorted(
            per_exp.items(),
            key=lambda kv: (
                kv[1]["mean_validation_acc"],
                kv[1]["mean_validation_auc"] if kv[1]["mean_validation_auc"] is not None else float("-inf"),
                -(kv[1]["mean_validation_rmse"] if kv[1]["mean_validation_rmse"] is not None else float("inf")),
            ),
            reverse=True,
        )
        print()
        print("===== Ranked By Mean Validation ACC Across DIRT_3/4 =====")
        for exp_name, stats in ranked_exp:
            print(
                f"{exp_name} | "
                f"mean_val_acc={format_value(stats['mean_validation_acc'])} | "
                f"mean_val_auc={format_value(stats['mean_validation_auc'])} | "
                f"mean_val_rmse={format_value(stats['mean_validation_rmse'])}"
            )


def parse_args():
    parser = argparse.ArgumentParser(description="Search best validation ACC on DIRT_3/4 only.")
    parser.add_argument("--mode", type=str, default="print", choices=["print", "run", "summary"])
    parser.add_argument("--python_bin", type=str, default="python")
    parser.add_argument("--script", type=str, default=DEFAULT_MODEL_SCRIPT)
    parser.add_argument("--data", type=str, default="assist2009")
    parser.add_argument("--cross_idx", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--stage1_epochs", type=int, default=10)
    parser.add_argument("--stage2_epochs", type=int, default=5)
    parser.add_argument("--attr_indices", type=str, default="3,4")
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--ws_root", type=str, default="ws/acc_search_dirt34")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    os.chdir(PROJECT_ROOT)
    if args.mode == "print":
        print_commands(args)
    elif args.mode == "run":
        run_commands(args)
    else:
        print_summaries(args)
