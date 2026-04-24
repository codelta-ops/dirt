import argparse
import json
import os
import subprocess
from typing import List


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE_SCRIPT = os.path.join("models", "dirt_baseline.py")
FULL_SCRIPT = os.path.join("models", "dirt_plus.py")


def parse_string_list(raw_value: str) -> List[str]:
    return [item.strip() for item in raw_value.split(",") if item.strip()]


def format_command(cmd: List[str]) -> str:
    return " ".join(cmd)


def has_metric_record(metrics_path: str, dtype: str) -> bool:
    if not os.path.exists(metrics_path):
        return False
    with open(metrics_path, "r", encoding="utf8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("dtype") == dtype:
                return True
    return False


def base_dataset_root(args, dataset: str) -> str:
    return os.path.join(args.base_ws_root, dataset, f"seed_{args.seed}", "base_model")


def base_metrics_path(args, dataset: str, attr_idx: int) -> str:
    return os.path.join(base_dataset_root(args, dataset), f"DIRT_{attr_idx}", "metrics.jsonl")


def full_metrics_path(args, dataset: str, attr_idx: int) -> str:
    return os.path.join(
        args.full_ws_root,
        dataset,
        f"seed_{args.seed}",
        "full_model",
        f"DIRT_{attr_idx}",
        "metrics.jsonl",
    )


def is_base_dataset_complete(args, dataset: str) -> bool:
    return all(has_metric_record(base_metrics_path(args, dataset, attr_idx), "test") for attr_idx in [1, 2, 3, 4])


def is_full_dataset_complete(args, dataset: str) -> bool:
    return all(has_metric_record(full_metrics_path(args, dataset, attr_idx), "final_summary") for attr_idx in [1, 2, 3, 4])


def build_base_command(args, dataset: str) -> List[str]:
    return [
        args.python_bin,
        BASE_SCRIPT,
        "--data", dataset,
        "--cross_idx", str(args.cross_idx),
        "--device", args.device,
        "--lr", f"{args.base_lr:g}",
        "--batch_size", str(args.batch_size),
        "--stu_ho_dim", str(args.stu_ho_dim),
        "--rnn_type", args.rnn_type,
        "--ws_root", base_dataset_root(args, dataset),
    ]


def build_full_command(args, dataset: str) -> List[str]:
    return [
        args.python_bin,
        FULL_SCRIPT,
        "--data", dataset,
        "--cross_idx", str(args.cross_idx),
        "--seed", str(args.seed),
        "--device", args.device,
        "--stage1_epochs", str(args.stage1_epochs),
        "--stage2_epochs", str(args.stage2_epochs),
        "--ws_root", os.path.join(args.full_ws_root, dataset, f"seed_{args.seed}"),
        "--exp_name", "full_model",
        "--attr_indices", args.attr_indices,
        "--loss_weight_mode", "learnable",
        "--step_weight_hidden_dim", "32",
        "--step_weight_dropout", "0.1",
        "--step_weight_use_teacher_confidence", "0",
        "--step_weight_use_position_feature", "1",
        "--step_weight_min", "0.8",
        "--step_weight_max", "2",
        "--stage1_lr", f"{args.full_stage1_lr:g}",
        "--stage2_lr", f"{args.full_stage2_lr:g}",
        "--lambda_consistency", f"{args.lambda_consistency:g}",
        "--consistency_warmup_epochs", str(args.consistency_warmup_epochs),
        "--use_temporal_self_attention", "1",
        "--use_query_guided_attention", "1",
        "--use_confidence_consistency", "1",
    ]


def print_commands(args) -> None:
    datasets = parse_string_list(args.datasets)
    for dataset in datasets:
        print(f"# Dataset: {dataset}")
        if args.run_base:
            cmd = build_base_command(args, dataset)
            print("# Base")
            print(format_command(cmd))
        if args.run_full:
            cmd = build_full_command(args, dataset)
            print("# Full")
            print(format_command(cmd))
        print()


def run_commands(args, skip_completed: bool = False) -> None:
    datasets = parse_string_list(args.datasets)
    for dataset in datasets:
        print(f">>> Dataset={dataset}")
        if args.run_base:
            if skip_completed and is_base_dataset_complete(args, dataset):
                print(">>> Skipping base (all DIRT_1~4 already completed)")
            else:
                cmd = build_base_command(args, dataset)
                print(">>> Running base")
                print(format_command(cmd))
                subprocess.run(cmd, check=True)
        if args.run_full:
            if skip_completed and is_full_dataset_complete(args, dataset):
                print(">>> Skipping full (all TIDE_1~4 already completed)")
            else:
                cmd = build_full_command(args, dataset)
                print(">>> Running full")
                print(format_command(cmd))
                subprocess.run(cmd, check=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Run only base and weighted full experiments for selected datasets.")
    parser.add_argument("--mode", type=str, default="print", choices=["print", "run", "resume"])
    parser.add_argument("--python_bin", type=str, default="python")
    parser.add_argument("--datasets", type=str, default="assist2009,assist2012,kddcup")
    parser.add_argument("--cross_idx", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--attr_indices", type=str, default="1,2,3,4")
    parser.add_argument("--run_base", action="store_true")
    parser.add_argument("--run_full", action="store_true")
    parser.add_argument("--base_ws_root", type=str, default="ws/chapter3_runs")
    parser.add_argument("--full_ws_root", type=str, default="ws/chapter3_runs")
    parser.add_argument("--base_lr", type=float, default=0.002)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--stu_ho_dim", type=int, default=50)
    parser.add_argument("--rnn_type", type=str, default="gru")
    parser.add_argument("--stage1_epochs", type=int, default=10)
    parser.add_argument("--stage2_epochs", type=int, default=5)
    parser.add_argument("--full_stage1_lr", type=float, default=0.002)
    parser.add_argument("--full_stage2_lr", type=float, default=0.002)
    parser.add_argument("--lambda_consistency", type=float, default=0.2)
    parser.add_argument("--consistency_warmup_epochs", type=int, default=4)
    args = parser.parse_args()
    if not args.run_base and not args.run_full:
        args.run_base = True
        args.run_full = True
    return args


if __name__ == "__main__":
    args = parse_args()
    os.chdir(PROJECT_ROOT)
    if args.mode == "print":
        print_commands(args)
    elif args.mode == "resume":
        run_commands(args, skip_completed=True)
    else:
        run_commands(args)
