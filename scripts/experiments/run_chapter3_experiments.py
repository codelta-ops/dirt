import argparse
import os
import subprocess
from typing import Dict, List


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_MODEL_SCRIPT = os.path.join("models", "dirt_plus.py")


EXPERIMENTS: List[Dict] = [
    {
        "name": "full_model",
        "purpose": "Final adopted DIRT+ setting for the main result.",
        "args": {
            "loss_weight_mode": "learnable",
            "step_weight_hidden_dim": 32,
            "step_weight_dropout": 0.10,
            "step_weight_use_teacher_confidence": 0,
            "step_weight_use_position_feature": 1,
            "step_weight_min": 0.8,
            "step_weight_max": 2.0,
            "stage1_lr": 0.002,
            "stage2_lr": 0.0006,
            "lambda_consistency": 0.2,
            "consistency_warmup_epochs": 4,
            "use_temporal_self_attention": 1,
            "use_query_guided_attention": 1,
            "use_confidence_consistency": 1,
        },
    },
    {
        "name": "wo_attention",
        "purpose": "Ablation without temporal self-attention.",
        "args": {
            "loss_weight_mode": "learnable",
            "step_weight_hidden_dim": 32,
            "step_weight_dropout": 0.10,
            "step_weight_use_teacher_confidence": 0,
            "step_weight_use_position_feature": 1,
            "step_weight_min": 0.8,
            "step_weight_max": 2.0,
            "stage1_lr": 0.002,
            "stage2_lr": 0.0006,
            "lambda_consistency": 0.2,
            "consistency_warmup_epochs": 4,
            "use_temporal_self_attention": 0,
            "use_query_guided_attention": 1,
            "use_confidence_consistency": 1,
        },
    },
    {
        "name": "wo_query",
        "purpose": "Ablation without query-guided aggregation.",
        "args": {
            "loss_weight_mode": "learnable",
            "step_weight_hidden_dim": 32,
            "step_weight_dropout": 0.10,
            "step_weight_use_teacher_confidence": 0,
            "step_weight_use_position_feature": 1,
            "step_weight_min": 0.8,
            "step_weight_max": 2.0,
            "stage1_lr": 0.002,
            "stage2_lr": 0.0006,
            "lambda_consistency": 0.2,
            "consistency_warmup_epochs": 4,
            "use_temporal_self_attention": 1,
            "use_query_guided_attention": 0,
            "use_confidence_consistency": 1,
        },
    },
    {
        "name": "wo_weight",
        "purpose": "Ablation without dynamic weighting, using plain BCE.",
        "args": {
            "loss_weight_mode": "plain",
            "stage1_lr": 0.002,
            "stage2_lr": 0.0006,
            "lambda_consistency": 0.2,
            "consistency_warmup_epochs": 4,
            "use_temporal_self_attention": 1,
            "use_query_guided_attention": 1,
            "use_confidence_consistency": 1,
        },
    },
    {
        "name": "wo_consistency",
        "purpose": "Ablation without stage2 confidence consistency.",
        "args": {
            "loss_weight_mode": "learnable",
            "step_weight_hidden_dim": 32,
            "step_weight_dropout": 0.10,
            "step_weight_use_teacher_confidence": 0,
            "step_weight_use_position_feature": 1,
            "step_weight_min": 0.8,
            "step_weight_max": 2.0,
            "stage1_lr": 0.002,
            "stage2_lr": 0.0006,
            "lambda_consistency": 0.0,
            "consistency_warmup_epochs": 4,
            "use_temporal_self_attention": 1,
            "use_query_guided_attention": 1,
            "use_confidence_consistency": 0,
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


def parse_args():
    parser = argparse.ArgumentParser(description="Run the minimum DIRT+ experiments needed for Chapter 3 tables.")
    parser.add_argument("--mode", type=str, default="print", choices=["print", "run"])
    parser.add_argument("--python_bin", type=str, default="python")
    parser.add_argument("--script", type=str, default=DEFAULT_MODEL_SCRIPT)
    parser.add_argument("--data", type=str, default="assist2009")
    parser.add_argument("--cross_idx", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--stage1_epochs", type=int, default=10)
    parser.add_argument("--stage2_epochs", type=int, default=5)
    parser.add_argument("--attr_indices", type=str, default="1,2,3,4")
    parser.add_argument("--ws_root", type=str, default="ws/chapter3_runs")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    os.chdir(PROJECT_ROOT)
    if args.mode == "print":
        print_commands(args)
    else:
        run_commands(args)
