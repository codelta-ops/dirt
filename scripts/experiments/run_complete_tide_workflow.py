import argparse
import os
import subprocess
from typing import List, Tuple


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESET_SCRIPT = os.path.join("scripts", "maintenance", "reset_local_experiment_outputs.py")
MAIN_SCRIPT = os.path.join("scripts", "experiments", "run_base_and_full.py")
ABLATION_SCRIPT = os.path.join("scripts", "experiments", "run_chapter3_experiments.py")
EXPORT_SCRIPT = os.path.join("scripts", "reports", "export_tide_final_tables.py")


def format_command(cmd: List[str]) -> str:
    return " ".join(cmd)


def script_mode(mode: str) -> str:
    if mode == "print":
        return "print"
    if mode == "resume":
        return "resume"
    return "run"


def build_reset_command(args) -> List[str]:
    return [
        args.python_bin,
        RESET_SCRIPT,
    ]


def build_main_command(args) -> List[str]:
    return [
        args.python_bin,
        MAIN_SCRIPT,
        "--mode",
        script_mode(args.mode),
        "--datasets",
        args.datasets,
        "--cross_idx",
        str(args.cross_idx),
        "--seed",
        str(args.seed),
        "--device",
        args.device,
        "--attr_indices",
        args.attr_indices,
        "--base_ws_root",
        args.base_ws_root,
        "--full_ws_root",
        args.full_ws_root,
        "--base_lr",
        f"{args.base_lr:g}",
        "--batch_size",
        str(args.batch_size),
        "--stu_ho_dim",
        str(args.stu_ho_dim),
        "--rnn_type",
        args.rnn_type,
        "--stage1_epochs",
        str(args.stage1_epochs),
        "--stage2_epochs",
        str(args.stage2_epochs),
        "--full_stage1_lr",
        f"{args.full_stage1_lr:g}",
        "--full_stage2_lr",
        f"{args.full_stage2_lr:g}",
        "--lambda_consistency",
        f"{args.lambda_consistency:g}",
        "--consistency_warmup_epochs",
        str(args.consistency_warmup_epochs),
        "--run_base",
        "--run_full",
    ]


def build_ablation_command(args) -> List[str]:
    return [
        args.python_bin,
        ABLATION_SCRIPT,
        "--mode",
        script_mode(args.mode),
        "--datasets",
        args.ablation_datasets,
        "--seed",
        str(args.seed),
        "--cross_idx",
        str(args.cross_idx),
        "--device",
        args.device,
        "--stage1_epochs",
        str(args.stage1_epochs),
        "--stage2_epochs",
        str(args.stage2_epochs),
        "--attr_indices",
        args.attr_indices,
        "--experiments",
        "wo_attention,wo_query,wo_weight,wo_consistency",
        "--ws_root",
        args.full_ws_root,
    ]


def build_export_command(args) -> List[str]:
    return [
        args.python_bin,
        EXPORT_SCRIPT,
        "--full_ws_root",
        args.full_ws_root,
        "--base_ws_root",
        args.base_ws_root,
        "--output_dir",
        args.output_dir,
        "--datasets",
        args.datasets,
        "--seed",
        str(args.seed),
    ]


def build_workflow_steps(args) -> List[Tuple[str, str, List[str]]]:
    steps: List[Tuple[str, str, List[str]]] = []
    if args.reset_outputs:
        steps.append(("reset", "Reset local experiment outputs under ws/", build_reset_command(args)))
    steps.append(("main", "Run three-dataset DIRT vs TIDE main experiments", build_main_command(args)))
    steps.append(("ablation", "Run ASSIST2009 ablation experiments", build_ablation_command(args)))
    steps.append(("export", "Export fixed final tables", build_export_command(args)))
    return steps


def print_workflow(args) -> None:
    steps = build_workflow_steps(args)
    for idx, (step_id, description, cmd) in enumerate(steps, start=1):
        print(f"# Step {idx}: {step_id}")
        print(f"# {description}")
        print(format_command(cmd))
        print()


def run_workflow(args) -> None:
    steps = build_workflow_steps(args)
    for idx, (step_id, description, cmd) in enumerate(steps, start=1):
        print(f">>> Step {idx}/{len(steps)}: {step_id}")
        print(f">>> {description}")
        print(format_command(cmd))
        subprocess.run(cmd, check=True)

    print(f">>> Workflow completed. Final tables are available under: {args.output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run the full TIDE paper workflow: optional ws reset, main results, "
            "ASSIST2009 ablations, and final table export."
        )
    )
    parser.add_argument("--mode", type=str, default="print", choices=["print", "run", "resume"])
    parser.add_argument("--python_bin", type=str, default="python")
    parser.add_argument("--datasets", type=str, default="assist2009,assist2012,kddcup")
    parser.add_argument("--ablation_datasets", type=str, default="assist2009")
    parser.add_argument("--cross_idx", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--attr_indices", type=str, default="1,2,3,4")
    parser.add_argument("--base_ws_root", type=str, default="ws/chapter3_runs")
    parser.add_argument("--full_ws_root", type=str, default="ws/chapter3_runs")
    parser.add_argument("--output_dir", type=str, default="reports/tide_final_tables")
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
    parser.add_argument("--reset_outputs", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    os.chdir(PROJECT_ROOT)
    if args.mode == "print":
        print_workflow(args)
    else:
        run_workflow(args)
