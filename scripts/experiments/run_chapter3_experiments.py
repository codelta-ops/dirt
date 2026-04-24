import argparse
import json
import os
import subprocess
from typing import Dict, List


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_MODEL_SCRIPT = os.path.join("models", "dirt_plus.py")


EXPERIMENTS: List[Dict] = [
    {
        "name": "full_model",
        "purpose": "Final adopted DIRT+ setting for the main result, with learnable dynamic step weighting.",
        "args": {
            "loss_weight_mode": "learnable",
            "step_weight_hidden_dim": 32,
            "step_weight_dropout": 0.10,
            "step_weight_use_teacher_confidence": 0,
            "step_weight_use_position_feature": 1,
            "step_weight_min": 0.8,
            "step_weight_max": 2.0,
            "stage1_lr": 0.002,
            "stage2_lr": 0.002,
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
            "stage2_lr": 0.002,
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
            "stage2_lr": 0.002,
            "lambda_consistency": 0.2,
            "consistency_warmup_epochs": 4,
            "use_temporal_self_attention": 1,
            "use_query_guided_attention": 0,
            "use_confidence_consistency": 1,
        },
    },
    {
        "name": "wo_weight",
        "purpose": "Ablation without dynamic step weighting.",
        "args": {
            "loss_weight_mode": "plain",
            "stage1_lr": 0.002,
            "stage2_lr": 0.002,
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
            "stage2_lr": 0.002,
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


def parse_attr_indices(attr_indices: str) -> List[int]:
    return [int(item.strip()) for item in attr_indices.split(",") if item.strip()]


def parse_string_list(raw_value: str) -> List[str]:
    return [item.strip() for item in raw_value.split(",") if item.strip()]


def parse_seed_list(raw_value: str) -> List[int]:
    return [int(item.strip()) for item in raw_value.split(",") if item.strip()]


def resolve_dataset_names(args) -> List[str]:
    if args.datasets:
        return parse_string_list(args.datasets)
    return [args.data]


def resolve_seed_values(args) -> List[int]:
    if args.seeds:
        return parse_seed_list(args.seeds)
    return [args.seed]


def build_run_ws_root(base_ws_root: str, dataset_name: str, seed: int, multi_run: bool) -> str:
    return os.path.join(base_ws_root, dataset_name, f"seed_{seed}")


def current_data(args) -> str:
    return getattr(args, "current_data", args.data)


def current_seed(args) -> int:
    return int(getattr(args, "current_seed", args.seed))


def current_cross_idx(args) -> int:
    return int(getattr(args, "current_cross_idx", args.cross_idx))


def current_stage1_epochs(args) -> int:
    return int(getattr(args, "current_stage1_epochs", args.stage1_epochs))


def current_stage2_epochs(args) -> int:
    return int(getattr(args, "current_stage2_epochs", args.stage2_epochs))


def current_ws_root(args) -> str:
    return getattr(args, "current_ws_root", args.ws_root)


def build_expected_ws_config(args, exp_config, attr_idx: int) -> Dict:
    stage1_lr = exp_config["args"].get("stage1_lr", 0.002)
    stage2_lr = exp_config["args"].get("stage2_lr", stage1_lr)
    expected = {
        "attr_idx": int(attr_idx),
        "data": current_data(args),
        "cross_idx": current_cross_idx(args),
        "stage1_epochs": current_stage1_epochs(args),
        "stage2_epochs": current_stage2_epochs(args),
        "seed": current_seed(args),
        "stage1_lr": float(stage1_lr),
        "stage2_lr": float(stage2_lr),
        "loss_weight_mode": exp_config["args"].get("loss_weight_mode", "rule"),
        "use_temporal_self_attention": int(exp_config["args"].get("use_temporal_self_attention", 1)),
        "use_query_guided_attention": int(exp_config["args"].get("use_query_guided_attention", 1)),
        "use_confidence_consistency": int(exp_config["args"].get("use_confidence_consistency", 1)),
        "lambda_consistency": float(exp_config["args"].get("lambda_consistency", 0.2)),
        "consistency_warmup_epochs": int(exp_config["args"].get("consistency_warmup_epochs", 3)),
    }
    optional_keys = [
        "step_weight_hidden_dim",
        "step_weight_dropout",
        "step_weight_use_teacher_confidence",
        "step_weight_use_position_feature",
        "step_weight_min",
        "step_weight_max",
    ]
    for key in optional_keys:
        if key in exp_config["args"]:
            expected[key] = exp_config["args"][key]
    return expected


def config_value_matches(key: str, expected_value, actual_config: Dict) -> bool:
    actual_value = actual_config.get(key)
    if actual_value == expected_value:
        return True

    # Older workspaces were created before these fields were written into model_config.txt.
    legacy_defaults = {
        "cross_idx": 0,
        "stage1_epochs": 10,
        "stage2_epochs": 5,
    }
    if actual_value is None and key in legacy_defaults and expected_value == legacy_defaults[key]:
        return True
    return False


def is_attr_complete(args, exp_config, attr_idx: int) -> bool:
    ws_root = current_ws_root(args)
    exp_name = exp_config["name"]
    metrics_path = os.path.join(ws_root, exp_name, f"DIRT_{attr_idx}", "metrics.jsonl")
    config_path = os.path.join(ws_root, exp_name, f"DIRT_{attr_idx}", "model_config.txt")
    if not os.path.exists(metrics_path):
        return False
    if not os.path.exists(config_path):
        return False
    try:
        with open(config_path, "r", encoding="utf8") as f:
            actual_config = json.load(f)
    except json.JSONDecodeError:
        return False

    expected_config = build_expected_ws_config(args, exp_config, attr_idx)
    for key, expected_value in expected_config.items():
        if not config_value_matches(key, expected_value, actual_config):
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
            if record.get("dtype") == "final_summary":
                return True
    return False


def get_pending_attr_indices(args, exp_config) -> List[int]:
    requested = parse_attr_indices(args.attr_indices)
    pending = []
    for attr_idx in requested:
        if not is_attr_complete(args, exp_config, attr_idx):
            pending.append(attr_idx)
    return pending


def build_command(args, exp_config, attr_indices_override=None):
    attr_indices = attr_indices_override if attr_indices_override is not None else parse_attr_indices(args.attr_indices)
    cmd = [
        args.python_bin,
        args.script,
        "--data", current_data(args),
        "--cross_idx", str(current_cross_idx(args)),
        "--device", args.device,
        "--stage1_epochs", str(current_stage1_epochs(args)),
        "--stage2_epochs", str(current_stage2_epochs(args)),
        "--seed", str(current_seed(args)),
        "--ws_root", current_ws_root(args),
        "--exp_name", exp_config["name"],
        "--attr_indices", ",".join(str(item) for item in attr_indices),
    ]
    for key, value in exp_config["args"].items():
        cmd.extend([f"--{key}", format_value(value)])
    return cmd


def build_label(exp_config):
    pieces = [f"{k}={format_value(v)}" for k, v in exp_config["args"].items()]
    return f'{exp_config["name"]} | ' + ", ".join(pieces)


def print_commands(args):
    dataset_names = resolve_dataset_names(args)
    seed_values = resolve_seed_values(args)
    multi_run = len(dataset_names) > 1 or len(seed_values) > 1

    for dataset_name in dataset_names:
        for seed in seed_values:
            args.current_data = dataset_name
            args.current_seed = seed
            args.current_cross_idx = args.cross_idx
            args.current_stage1_epochs = args.stage1_epochs
            args.current_stage2_epochs = args.stage2_epochs
            args.current_ws_root = build_run_ws_root(args.ws_root, dataset_name, seed, multi_run)
            print(
                f"# Dataset: {args.current_data}, Seed: {args.current_seed}, "
                f"Workspace: {args.current_ws_root}"
            )
            for exp in EXPERIMENTS:
                print(f"# Experiment: {build_label(exp)}")
                print(f"# Purpose: {exp['purpose']}")
                print(" ".join(build_command(args, exp)))
                print()


def run_commands(args, skip_completed=False):
    dataset_names = resolve_dataset_names(args)
    seed_values = resolve_seed_values(args)
    multi_run = len(dataset_names) > 1 or len(seed_values) > 1

    for dataset_name in dataset_names:
        for seed in seed_values:
            args.current_data = dataset_name
            args.current_seed = seed
            args.current_cross_idx = args.cross_idx
            args.current_stage1_epochs = args.stage1_epochs
            args.current_stage2_epochs = args.stage2_epochs
            args.current_ws_root = build_run_ws_root(args.ws_root, dataset_name, seed, multi_run)

            print(
                f">>> Dataset={args.current_data}, Seed={args.current_seed}, "
                f"Workspace={args.current_ws_root}"
            )
            for idx, exp in enumerate(EXPERIMENTS, start=1):
                attr_indices = parse_attr_indices(args.attr_indices)
                if skip_completed:
                    attr_indices = get_pending_attr_indices(args, exp)
                    if not attr_indices:
                        print(
                            f">>> Skipping [{idx}/{len(EXPERIMENTS)}] {exp['name']} "
                            "(all requested attr_indices already completed)"
                        )
                        continue
                cmd = build_command(args, exp, attr_indices_override=attr_indices)
                print(f">>> Running [{idx}/{len(EXPERIMENTS)}] {build_label(exp)}")
                print(f">>> Purpose: {exp['purpose']}")
                print(f">>> Attr indices: {','.join(str(item) for item in attr_indices)}")
                print(" ".join(cmd))
                subprocess.run(cmd, check=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Run the minimum DIRT+ experiments needed for Chapter 3 tables.")
    parser.add_argument("--mode", type=str, default="print", choices=["print", "run", "resume"])
    parser.add_argument("--python_bin", type=str, default="python")
    parser.add_argument("--script", type=str, default=DEFAULT_MODEL_SCRIPT)
    parser.add_argument("--data", type=str, default="assist2009")
    parser.add_argument("--datasets", type=str, default=None)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--seeds", type=str, default=None)
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
    elif args.mode == "resume":
        run_commands(args, skip_completed=True)
    else:
        run_commands(args)
