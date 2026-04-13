import csv
import json
import os
from typing import Dict, List, Optional, Tuple


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "reports", "chapter3_tables")
DATASET_NAME = "assist2009"


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
        cfg = eval(f.read().strip())

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


def scan_workspace_files() -> List[str]:
    found = []
    for root, _, files in os.walk(PROJECT_ROOT):
        if "metrics.jsonl" in files:
            found.append(root)
    return sorted(found)


def collect_stage_rows_from_logs() -> List[Dict]:
    rows: List[Dict] = []
    for ws_dir in scan_workspace_files():
        ws_name = os.path.basename(ws_dir)
        if ws_name not in {"DIRT_3", "DIRT_4"}:
            continue
        records = read_jsonl(os.path.join(ws_dir, "metrics.jsonl"))
        for stage in [1, 2]:
            best_val = find_best_validation_record(records, stage)
            if best_val is None:
                continue
            test_record = find_test_record(records, stage, int(best_val["epoch"]))
            rows.append(
                {
                    "Model": ws_name,
                    "Stage": stage,
                    "Validation AUC": float(best_val.get("auc", 0.0)),
                    "Validation ACC": float(best_val.get("acc", 0.0)),
                    "Validation RMSE": float(best_val.get("rmse", 0.0)),
                    "Test AUC": float(test_record.get("auc", 0.0)) if test_record else "N/A",
                    "Test ACC": float(test_record.get("acc", 0.0)) if test_record else "N/A",
                    "Test RMSE": float(test_record.get("rmse", 0.0)) if test_record else "N/A",
                    "Best Epoch": int(best_val.get("epoch", -1)),
                }
            )
    return rows


def collect_dirt_plus_default_result() -> Dict:
    for ws_dir in scan_workspace_files():
        cfg_path = os.path.join(ws_dir, "model_config.txt")
        if not os.path.exists(cfg_path):
            continue
        cfg = read_json(cfg_path)
        if (
            cfg.get("loss_weight_mode") == "rule"
            and int(cfg.get("use_adaptive_fusion", 0)) == 1
            and int(cfg.get("use_temporal_bias", 0)) == 1
            and int(cfg.get("use_exercise_aware_decay", 0)) == 1
        ):
            records = read_jsonl(os.path.join(ws_dir, "metrics.jsonl"))
            tests = [r for r in records if r.get("dtype") == "test"]
            if not tests:
                continue
            last = tests[-1]
            return {
                "Model": "DIRT+ default configuration",
                "AUC": float(last.get("auc", 0.0)),
                "ACC": float(last.get("acc", 0.0)),
                "RMSE": float(last.get("rmse", 0.0)),
            }
    return {"Model": "DIRT+ default configuration", "AUC": "N/A", "ACC": "N/A", "RMSE": "N/A"}


def main_results_table() -> List[Dict]:
    models = [
        "IRT",
        "DKT",
        "SAKT",
        "AKT",
        "Original DIRT",
        "Normalized DIRT",
    ]
    rows = [{"Model": m, "AUC": "N/A", "ACC": "N/A", "RMSE": "N/A"} for m in models]
    rows.append(collect_dirt_plus_default_result())
    return rows


def ablation_table() -> List[Dict]:
    variants = [
        "Full Model",
        "w/o attention",
        "w/o query",
        "w/o weight",
        "w/o consistency",
        "w/o adaptive fusion",
        "w/o temporal bias",
        "w/o exercise-aware decay",
    ]
    return [{"Variant": v, "AUC": "N/A", "ACC": "N/A", "RMSE": "N/A"} for v in variants]


def stage_comparison_table() -> List[Dict]:
    rows = collect_stage_rows_from_logs()
    if rows:
        return rows
    rows = []
    for model in ["DIRT_3", "DIRT_4"]:
        rows.append(
            {
                "Model": model,
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


def training_config_table() -> List[Dict]:
    return [
        {"Setting": "batch size", "Value": 32},
        {"Setting": "stage1 epochs", "Value": 10},
        {"Setting": "stage2 epochs", "Value": 5},
        {"Setting": "stage1 lr", "Value": 0.002},
        {"Setting": "stage2 lr", "Value": 0.0008},
        {"Setting": "optimizer", "Value": "Adam"},
        {"Setting": "grad clip", "Value": 5.0},
        {"Setting": "scheduler", "Value": "ReduceLROnPlateau (monitor validation AUC)"},
        {"Setting": "seed", "Value": 2024},
        {"Setting": "loss_weight_mode", "Value": "learnable (adopted control group)"},
        {"Setting": "use_adaptive_fusion", "Value": 1},
        {"Setting": "use_temporal_bias", "Value": 1},
        {"Setting": "use_exercise_aware_decay", "Value": 1},
        {"Setting": "use_confidence_consistency", "Value": 1},
        {"Setting": "use_state_consistency", "Value": 0},
    ]


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


def export_tables() -> None:
    ensure_dir(OUTPUT_DIR)
    tables = [
        ("dataset_statistics", dataset_stats()),
        ("main_results", main_results_table()),
        ("ablation_results", ablation_table()),
        ("stage_comparison", stage_comparison_table()),
        ("training_configuration", training_config_table()),
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
        csv_path = os.path.join(OUTPUT_DIR, f"{name}.csv")
        write_csv(csv_path, fieldnames, rows)
        md_sections.append(f"## {name}")
        md_sections.append("")
        md_sections.append(markdown_table(fieldnames, rows))
        md_sections.append("")

    md_sections.append("## Suggested Placement")
    md_sections.append("")
    md_sections.extend(placement_lines)
    md_sections.append("")

    with open(os.path.join(OUTPUT_DIR, "chapter3_tables.md"), "w", encoding="utf8") as f:
        f.write("\n".join(md_sections))


if __name__ == "__main__":
    export_tables()
