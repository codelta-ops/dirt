import ast
import argparse
import csv
import json
import os
import sys
from typing import Dict, List, Optional, Tuple


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "reports", "chapter3_tables")
DATASET_NAME = "assist2009"
DEFAULT_WS_ROOT = os.path.join(PROJECT_ROOT, "ws", "chapter3_runs")

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.reports.export_paper_stats import (  # noqa: E402
    build_ablation_table,
    build_main_results_table,
    build_stage_comparison_table,
    build_training_config_table,
    collect_experiment_results,
    parse_markdown_tables,
)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def read_json(path: str):
    with open(path, "r", encoding="utf8") as f:
        return json.load(f)


def read_jsonl(path: str) -> List[Dict]:
    rows: List[Dict] = []
    with open(path, "r", encoding="utf8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def fmt(value) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def write_csv(path: str, fieldnames: List[str], rows: List[Dict]) -> None:
    with open(path, "w", encoding="utf8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "N/A") for k in fieldnames})


def markdown_table(fieldnames: List[str], rows: List[Dict]) -> str:
    lines = []
    lines.append("| " + " | ".join(fieldnames) + " |")
    lines.append("| " + " | ".join(["---"] * len(fieldnames)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(k, "N/A")) for k in fieldnames) + " |")
    return "\n".join(lines)


def dataset_stats() -> List[Dict]:
    with open(os.path.join(PROJECT_ROOT, "data", DATASET_NAME, "data_config.txt"), "r", encoding="utf8") as f:
        cfg = ast.literal_eval(f.read().strip())

    split_files = ["train_0.json", "val_0.json", "test.json"]
    total_students = 0
    total_interactions = 0
    total_correct = 0
    max_length = 0

    for name in split_files:
        path = os.path.join(PROJECT_ROOT, "data", DATASET_NAME, name)
        data = read_json(path)
        total_students += len(data)
        for stu in data:
            log_len = int(stu[1])
            total_interactions += log_len
            max_length = max(max_length, log_len)
            for i in range(log_len):
                total_correct += int(stu[2][i][2])

    avg_length = total_interactions / total_students if total_students else None
    correct_rate = total_correct / total_interactions if total_interactions else None

    return [
        {
            "Dataset": DATASET_NAME,
            "#Students": int(cfg["student_n"]),
            "#Exercises": int(cfg["exer_n"]),
            "#Concepts": int(cfg["knowledge_n"]),
            "#Interactions": int(total_interactions),
            "Correct Rate": correct_rate,
            "Avg Length": avg_length,
            "Max Length": int(max_length),
        }
    ]


def validation_priority_key(record: Dict) -> Tuple[float, float, float, int]:
    return (
        float(record.get("auc", 0.0)),
        float(record.get("acc", 0.0)),
        -float(record.get("rmse", float("inf"))),
        -int(record.get("epoch", 0)),
    )


def find_best_validation_record(records: List[Dict], stage: int) -> Optional[Dict]:
    candidates = [r for r in records if r.get("dtype") == "validation" and int(r.get("stage", -1)) == int(stage)]
    if not candidates:
        return None
    return max(candidates, key=validation_priority_key)


def find_test_record(records: List[Dict], stage: int, epoch: int) -> Optional[Dict]:
    for r in records:
        if r.get("dtype") == "test" and int(r.get("stage", -1)) == int(stage) and int(r.get("epoch", -1)) == int(epoch):
            return r
    return None


def latest_final_summary(records: List[Dict]) -> Optional[Dict]:
    summaries = [r for r in records if r.get("dtype") == "final_summary"]
    if not summaries:
        return None
    return summaries[-1]


def main_results_table(results: Dict[str, Dict[str, Dict]], manuscript_md: str) -> List[Dict]:
    manuscript_tables = parse_markdown_tables(manuscript_md)
    rows = build_main_results_table(results, manuscript_tables)
    rmse_by_model = {}
    full = results.get("full_model", {})
    for attr_name in ["DIRT_1", "DIRT_2", "DIRT_3", "DIRT_4"]:
        attr = full.get(attr_name, {})
        if not attr.get("has_final_summary"):
            continue
        rmse_by_model[f"DIRT+_{attr_name.split('_')[1]}"] = attr["final"].get("Test RMSE", "N/A")
    for row in rows:
        row["RMSE"] = rmse_by_model.get(row.get("Model"), row.get("RMSE", "N/A"))
    return rows


def ablation_table(results: Dict[str, Dict[str, Dict]]) -> List[Dict]:
    rows = build_ablation_table(results)
    variants = {row["Variant"] for row in rows}
    for variant in ["w/o adaptive fusion", "w/o temporal bias", "w/o exercise-aware decay"]:
        if variant not in variants:
            rows.append({"Variant": variant, "AUC": "N/A", "ACC": "N/A", "RMSE": "N/A"})
    return rows


def stage_comparison_table(results: Dict[str, Dict[str, Dict]]) -> List[Dict]:
    rows = build_stage_comparison_table(results)
    if rows:
        return rows
    rows = []
    for model in ["DIRT_3", "DIRT_4"]:
        rows.append(
            {
                "Model": f"DIRT+_{model.split('_')[1]}",
                "Stage": "N/A",
                "Validation AUC": "N/A",
                "Validation ACC": "N/A",
                "Validation RMSE": "N/A",
                "Test AUC": "N/A",
                "Test ACC": "N/A",
                "Test RMSE": "N/A",
                "Best Epoch": "N/A",
            }
        )
    return rows


def training_config_table(results: Dict[str, Dict[str, Dict]]) -> List[Dict]:
    rows = build_training_config_table(results)
    renamed = []
    for row in rows:
        setting = row["Setting"]
        value = row["Value"]
        if setting == "Batch size":
            renamed.append({"Setting": "batch size", "Value": value})
        elif setting == "Epochs (Stage1 / Stage2)":
            stage1_epochs, stage2_epochs = [item.strip() for item in str(value).split("/", 1)]
            renamed.append({"Setting": "stage1 epochs", "Value": stage1_epochs})
            renamed.append({"Setting": "stage2 epochs", "Value": stage2_epochs})
        elif setting == "Learning rate":
            parts = [item.strip() for item in str(value).split(",")]
            stage1_lr = parts[0].split("=", 1)[1].strip() if len(parts) > 0 and "=" in parts[0] else value
            stage2_lr = parts[1].split("=", 1)[1].strip() if len(parts) > 1 and "=" in parts[1] else value
            renamed.append({"Setting": "stage1 lr", "Value": stage1_lr})
            renamed.append({"Setting": "stage2 lr", "Value": stage2_lr})
        elif setting == "Optimizer":
            renamed.append({"Setting": "optimizer", "Value": value})
        elif setting == "Seed":
            renamed.append({"Setting": "seed", "Value": value})
        elif setting == "Dynamic weight":
            renamed.append({"Setting": "loss_weight_mode", "Value": value})
        elif setting == "Consistency":
            renamed.append({"Setting": "use_confidence_consistency", "Value": value})
        else:
            renamed.append({"Setting": setting, "Value": value})
    renamed.extend(
        [
            {"Setting": "grad clip", "Value": 5.0},
            {"Setting": "scheduler", "Value": "ReduceLROnPlateau (monitor validation AUC)"},
            {"Setting": "use_adaptive_fusion", "Value": 1},
            {"Setting": "use_temporal_bias", "Value": 1},
            {"Setting": "use_exercise_aware_decay", "Value": 1},
            {"Setting": "use_state_consistency", "Value": 0},
        ]
    )
    return renamed


def mechanism_analysis_table() -> List[Dict]:
    return [
        {
            "step_weight_mean": "N/A",
            "step_weight_std": "N/A",
            "step_weight_min": "N/A",
            "step_weight_max": "N/A",
            "teacher_confidence_mean": "N/A",
            "normalized_confidence_mean": "N/A",
            "consistency_loss": "N/A",
            "fusion_gate_rnn_mean": "N/A",
            "fusion_gate_attn_mean": "N/A",
            "fusion_gate_query_mean": "N/A",
            "decay_gamma_mean": "N/A",
            "decay_gamma_std": "N/A",
        }
    ]


def multi_seed_table() -> List[Dict]:
    return [
        {"Model": "DIRT+ default configuration", "AUC(mean±std)": "N/A", "ACC(mean±std)": "N/A", "RMSE(mean±std)": "N/A"}
    ]


def export_tables(ws_root: str, output_dir: str, manuscript_md: str) -> None:
    ensure_dir(output_dir)
    results = collect_experiment_results(ws_root)
    tables = [
        ("dataset_statistics", dataset_stats()),
        ("main_results", main_results_table(results, manuscript_md)),
        ("ablation_results", ablation_table(results)),
        ("stage_comparison", stage_comparison_table(results)),
        ("training_configuration", training_config_table(results)),
        ("mechanism_analysis", mechanism_analysis_table()),
        ("multi_seed_stability", multi_seed_table()),
    ]

    md_sections = ["# Chapter 3 Tables", "", "This export uses local files found in the current workspace snapshot. Missing experiment logs are filled as `N/A`.", ""]
    placement_lines = [
        "- Dataset statistics: 3.1 Dataset and Preprocessing",
        "- Main results: 3.3 Overall Performance Comparison",
        "- Ablation results: 3.4 Ablation Study",
        "- Stage comparison: 3.5 Stage-wise Analysis",
        "- Training configuration: 3.2 Experimental Setup",
        "- Mechanism analysis: 3.6 Mechanism Analysis / Interpretability",
        "- Multi-seed stability: 3.7 Stability Analysis",
    ]

    for name, rows in tables:
        fieldnames = list(rows[0].keys()) if rows else []
        csv_path = os.path.join(output_dir, f"{name}.csv")
        write_csv(csv_path, fieldnames, rows)
        md_sections.append(f"## {name}")
        md_sections.append("")
        md_sections.append(markdown_table(fieldnames, rows))
        md_sections.append("")

    md_sections.append("## Suggested Placement")
    md_sections.append("")
    md_sections.extend(placement_lines)
    md_sections.append("")

    with open(os.path.join(output_dir, "chapter3_tables.md"), "w", encoding="utf8") as f:
        f.write("\n".join(md_sections))


def parse_args():
    parser = argparse.ArgumentParser(description="Export broad Chapter 3 tables from current experiment logs.")
    parser.add_argument("--ws_root", type=str, default=DEFAULT_WS_ROOT)
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--manuscript_md", type=str, default="")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    export_tables(
        ws_root=args.ws_root,
        output_dir=args.output_dir,
        manuscript_md=args.manuscript_md,
    )
