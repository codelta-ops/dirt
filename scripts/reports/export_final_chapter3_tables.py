import csv
import os
from typing import Dict, List


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "reports", "chapter3_final_tables")
DATASET_NAME = "assist2009"


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


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
    lines = [
        "| " + " | ".join(fieldnames) + " |",
        "| " + " | ".join(["---"] * len(fieldnames)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(k, "N/A")) for k in fieldnames) + " |")
    return "\n".join(lines)


def load_dataset_stats() -> Dict:
    cfg_path = os.path.join(PROJECT_ROOT, "data", DATASET_NAME, "data_config.txt")
    with open(cfg_path, "r", encoding="utf8") as f:
        cfg = eval(f.read().strip())

    split_files = ["train_0.json", "val_0.json", "test.json"]
    students = 0
    interactions = 0
    correct = 0
    for name in split_files:
        path = os.path.join(PROJECT_ROOT, "data", DATASET_NAME, name)
        import json
        with open(path, "r", encoding="utf8") as f:
            data = json.load(f)
        students += len(data)
        for stu in data:
            log_len = int(stu[1])
            interactions += log_len
            for i in range(log_len):
                correct += int(stu[2][i][2])

    return {
        "Dataset": DATASET_NAME,
        "#Students": int(cfg["student_n"]),
        "#Exercises": int(cfg["exer_n"]),
        "#Concepts": int(cfg["knowledge_n"]),
        "#Interactions": int(interactions),
        "Correct Rate": correct / interactions if interactions else "N/A",
    }


def table_main_results() -> List[Dict]:
    models = [
        "BKT",
        "DKVMN",
        "DKT_Q",
        "DKT_KC",
        "DKT_MLP",
        "DIRT_1",
        "DIRT_2",
        "DIRT_3",
        "DIRT_4",
        "DNeuralCDM_1",
        "DNeuralCDM_2",
        "DNeuralCDM_3",
        "DNeuralCDM_4",
        "DIRT+_3",
        "DIRT+_4",
        "DIRT+_1",
        "DIRT+_2",
    ]
    return [{"Model": m, "AUC": "N/A", "ACC": "N/A", "RMSE": "N/A"} for m in models]


def table_ablation() -> List[Dict]:
    variants = [
        "Full Model",
        "w/o attention",
        "w/o query",
        "w/o weight",
        "w/o consistency",
    ]
    return [{"Variant": v, "AUC": "N/A", "ACC": "N/A", "RMSE": "N/A"} for v in variants]


def table_stage_comparison() -> List[Dict]:
    rows = []
    for model in ["DIRT+_3", "DIRT+_4"]:
        for stage in ["stage1", "stage2"]:
            rows.append({"Model": model, "Stage": stage, "AUC": "N/A", "ACC": "N/A", "RMSE": "N/A"})
    return rows


def table_dataset() -> List[Dict]:
    return [load_dataset_stats()]


def table_training_config() -> List[Dict]:
    return [
        {"Setting": "batch size", "Value": 32},
        {"Setting": "epochs (stage1 / stage2)", "Value": "10 / 5"},
        {"Setting": "learning rate", "Value": "stage1=0.002, stage2=0.0008"},
        {"Setting": "optimizer", "Value": "Adam"},
        {"Setting": "seed", "Value": 2024},
        {"Setting": "attention", "Value": "Yes"},
        {"Setting": "query", "Value": "Yes"},
        {"Setting": "dynamic weight", "Value": "Yes"},
        {"Setting": "consistency", "Value": "Yes"},
    ]


def export_all() -> None:
    ensure_dir(OUTPUT_DIR)
    tables = {
        "table1_main_results": table_main_results(),
        "table2_ablation": table_ablation(),
        "table3_stage_comparison": table_stage_comparison(),
        "table4_dataset_statistics": table_dataset(),
        "table5_training_config": table_training_config(),
    }

    md_lines: List[str] = []
    for title, rows in tables.items():
        fieldnames = list(rows[0].keys())
        write_csv(os.path.join(OUTPUT_DIR, f"{title}.csv"), fieldnames, rows)
        md_lines.append(f"## {title}")
        md_lines.append("")
        md_lines.append(markdown_table(fieldnames, rows))
        md_lines.append("")

    with open(os.path.join(OUTPUT_DIR, "chapter3_final_tables.md"), "w", encoding="utf8") as f:
        f.write("\n".join(md_lines))


if __name__ == "__main__":
    export_all()
