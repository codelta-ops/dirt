import ast
import argparse
import csv
import json
import os
from typing import Dict, List, Optional, Tuple


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATASET_NAME = "assist2009"

EXPERIMENT_NAME_TO_VARIANT = {
    "full_model": "Full Model",
    "wo_attention": "w/o attention",
    "wo_query": "w/o query",
    "wo_consistency": "w/o consistency",
}

BASELINE_MODELS = [
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
]


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


def validation_priority_key(record: Dict) -> Tuple[float, float, float, int]:
    return (
        float(record.get("auc", 0.0)),
        float(record.get("acc", 0.0)),
        -float(record.get("rmse", float("inf"))),
        -int(record.get("epoch", 0)),
    )


def best_validation_record(records: List[Dict], stage: int) -> Optional[Dict]:
    candidates = [r for r in records if r.get("dtype") == "validation" and int(r.get("stage", -1)) == int(stage)]
    if not candidates:
        return None
    return max(candidates, key=validation_priority_key)


def test_record_for_epoch(records: List[Dict], stage: int, epoch: int) -> Optional[Dict]:
    for r in records:
        if r.get("dtype") == "test" and int(r.get("stage", -1)) == int(stage) and int(r.get("epoch", -1)) == int(epoch):
            return r
    return None


def latest_final_summary(records: List[Dict]) -> Optional[Dict]:
    summaries = [r for r in records if r.get("dtype") == "final_summary"]
    if not summaries:
        return None
    return summaries[-1]


def aggregate_mean(rows: List[Dict], fields: List[str]) -> Dict:
    result = {}
    for field in fields:
        vals = [float(r[field]) for r in rows if r.get(field) not in [None, "N/A"]]
        result[field] = sum(vals) / len(vals) if vals else "N/A"
    return result


def dataset_stats_row() -> Dict:
    cfg_path = os.path.join(PROJECT_ROOT, "data", DATASET_NAME, "data_config.txt")
    with open(cfg_path, "r", encoding="utf8") as f:
        cfg = ast.literal_eval(f.read().strip())

    interactions = 0
    correct = 0
    for split in ["train_0.json", "val_0.json", "test.json"]:
        path = os.path.join(PROJECT_ROOT, "data", DATASET_NAME, split)
        data = read_json(path)
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


def collect_experiment_results(ws_root: str) -> Dict[str, Dict[str, Dict]]:
    results: Dict[str, Dict[str, Dict]] = {}
    if not os.path.exists(ws_root):
        return results

    for exp_name in os.listdir(ws_root):
        exp_dir = os.path.join(ws_root, exp_name)
        if not os.path.isdir(exp_dir):
            continue
        attr_map: Dict[str, Dict] = {}
        for attr_dir_name in ["DIRT_1", "DIRT_2", "DIRT_3", "DIRT_4"]:
            attr_dir = os.path.join(exp_dir, attr_dir_name)
            metrics_path = os.path.join(attr_dir, "metrics.jsonl")
            cfg_path = os.path.join(attr_dir, "model_config.txt")
            if not os.path.exists(metrics_path):
                continue
            records = read_jsonl(metrics_path)
            cfg = read_json(cfg_path) if os.path.exists(cfg_path) else {}
            final_summary = latest_final_summary(records)

            if attr_dir_name in {"DIRT_1", "DIRT_2"}:
                stages = [1]
            else:
                stages = [1, 2]

            stage_rows = {}
            for stage in stages:
                best_val = best_validation_record(records, stage)
                if best_val is None:
                    continue
                test_row = test_record_for_epoch(records, stage, int(best_val["epoch"]))
                stage_rows[f"stage{stage}"] = {
                    "Validation AUC": float(best_val.get("auc", 0.0)),
                    "Validation ACC": float(best_val.get("acc", 0.0)),
                    "Validation RMSE": float(best_val.get("rmse", 0.0)),
                    "Best Epoch": int(best_val.get("epoch", -1)),
                    "Test AUC": float(test_row.get("auc", 0.0)) if test_row else "N/A",
                    "Test ACC": float(test_row.get("acc", 0.0)) if test_row else "N/A",
                    "Test RMSE": float(test_row.get("rmse", 0.0)) if test_row else "N/A",
                }

            final_stage = None
            final = {}
            if final_summary is not None:
                best_stage = int(final_summary.get("best_stage", final_summary.get("stage", -1)))
                final_stage = f"stage{best_stage}"
                final = {
                    "Validation AUC": float(final_summary.get("validation_auc", 0.0)),
                    "Validation ACC": float(final_summary.get("validation_acc", 0.0)),
                    "Validation RMSE": float(final_summary.get("validation_rmse", 0.0)),
                    "Best Epoch": int(final_summary.get("best_epoch", -1)),
                    "Test AUC": float(final_summary.get("test_auc", 0.0)),
                    "Test ACC": float(final_summary.get("test_acc", 0.0)),
                    "Test RMSE": float(final_summary.get("test_rmse", 0.0)),
                }
            attr_map[attr_dir_name] = {
                "config": cfg,
                "stages": stage_rows,
                "final_stage": final_stage,
                "has_final_summary": final_summary is not None,
                "final": final,
            }
        if attr_map:
            results[exp_name] = attr_map
    return results


def table1_main_results(results: Dict[str, Dict[str, Dict]]) -> List[Dict]:
    rows = [{"Model": model, "AUC": "N/A", "ACC": "N/A", "RMSE": "N/A"} for model in BASELINE_MODELS]
    full = results.get("full_model", {})
    for attr_name in ["DIRT_1", "DIRT_2", "DIRT_3", "DIRT_4"]:
        if attr_name not in full:
            continue
        if not full[attr_name].get("has_final_summary"):
            continue
        final = full[attr_name]["final"]
        rows.append(
            {
                "Model": f"DIRT+_{attr_name.split('_')[1]}",
                "AUC": final.get("Test AUC", "N/A"),
                "ACC": final.get("Test ACC", "N/A"),
                "RMSE": final.get("Test RMSE", "N/A"),
            }
        )
    return rows


def table2_ablation(results: Dict[str, Dict[str, Dict]]) -> List[Dict]:
    rows = []
    for exp_name, variant_name in EXPERIMENT_NAME_TO_VARIANT.items():
        exp = results.get(exp_name, {})
        final_rows = []
        for attr_name in ["DIRT_3", "DIRT_4"]:
            if attr_name in exp and exp[attr_name].get("has_final_summary") and exp[attr_name]["final"]:
                final_rows.append(exp[attr_name]["final"])
        if not final_rows:
            rows.append({"Variant": variant_name, "AUC": "N/A", "ACC": "N/A", "RMSE": "N/A"})
            continue
        agg = aggregate_mean(final_rows, ["Test AUC", "Test ACC", "Test RMSE"])
        rows.append(
            {
                "Variant": variant_name,
                "AUC": agg["Test AUC"],
                "ACC": agg["Test ACC"],
                "RMSE": agg["Test RMSE"],
            }
        )
    return rows


def table3_stage_comparison(results: Dict[str, Dict[str, Dict]]) -> List[Dict]:
    rows = []
    full = results.get("full_model", {})
    for attr_name in ["DIRT_3", "DIRT_4"]:
        attr = full.get(attr_name, {})
        if not attr.get("has_final_summary"):
            continue
        for stage_name in ["stage1", "stage2"]:
            stage_row = attr.get("stages", {}).get(stage_name, {})
            rows.append(
                {
                    "Model": f"DIRT+_{attr_name.split('_')[1]}",
                    "Stage": stage_name,
                    "Validation AUC": stage_row.get("Validation AUC", "N/A"),
                    "Validation ACC": stage_row.get("Validation ACC", "N/A"),
                    "Validation RMSE": stage_row.get("Validation RMSE", "N/A"),
                    "Test AUC": stage_row.get("Test AUC", "N/A"),
                    "Test ACC": stage_row.get("Test ACC", "N/A"),
                    "Test RMSE": stage_row.get("Test RMSE", "N/A"),
                    "Best Epoch": stage_row.get("Best Epoch", "N/A"),
                }
            )
    return rows


def table4_dataset_statistics() -> List[Dict]:
    return [dataset_stats_row()]


def table5_training_config(results: Dict[str, Dict[str, Dict]]) -> List[Dict]:
    cfg = {}
    full = results.get("full_model", {})
    for attr_name in ["DIRT_4", "DIRT_3", "DIRT_1", "DIRT_2"]:
        if attr_name in full:
            cfg = full[attr_name].get("config", {})
            break
    dynamic_weight_value = cfg.get("loss_weight_mode", "N/A")
    if dynamic_weight_value == "plain":
        dynamic_weight_value = "No"

    return [
        {"Setting": "batch size", "Value": cfg.get("batch_size", "N/A")},
        {"Setting": "epochs (stage1 / stage2)", "Value": f"{cfg.get('stage1_epochs', 10)} / {cfg.get('stage2_epochs', 5)}" if cfg else "N/A"},
        {"Setting": "learning rate", "Value": f"stage1={cfg.get('stage1_lr', 'N/A')}, stage2={cfg.get('stage2_lr', 'N/A')}"},
        {"Setting": "optimizer", "Value": "Adam"},
        {"Setting": "seed", "Value": cfg.get("seed", "N/A")},
        {"Setting": "attention", "Value": "Yes" if int(cfg.get("use_temporal_self_attention", 1)) == 1 else "No"},
        {"Setting": "query", "Value": "Yes" if int(cfg.get("use_query_guided_attention", 1)) == 1 else "No"},
        {"Setting": "dynamic weight", "Value": dynamic_weight_value},
        {"Setting": "consistency", "Value": "Yes" if int(cfg.get("use_confidence_consistency", 0)) == 1 else "No"},
    ]


def export_tables(ws_root: str, output_dir: str) -> None:
    ensure_dir(output_dir)
    results = collect_experiment_results(ws_root)
    tables = {
        "table1_main_results": table1_main_results(results),
        "table2_ablation": table2_ablation(results),
        "table3_stage_comparison": table3_stage_comparison(results),
        "table4_dataset_statistics": table4_dataset_statistics(),
        "table5_training_config": table5_training_config(results),
    }

    md_sections: List[str] = []
    for name, rows in tables.items():
        fieldnames = list(rows[0].keys())
        write_csv(os.path.join(output_dir, f"{name}.csv"), fieldnames, rows)
        md_sections.append(f"## {name}")
        md_sections.append("")
        md_sections.append(markdown_table(fieldnames, rows))
        md_sections.append("")

    with open(os.path.join(output_dir, "chapter3_collected_tables.md"), "w", encoding="utf8") as f:
        f.write("\n".join(md_sections))


def parse_args():
    parser = argparse.ArgumentParser(description="Collect final Chapter 3 tables from DIRT+ experiment logs.")
    parser.add_argument("--ws_root", type=str, default="ws/chapter3_runs")
    parser.add_argument("--output_dir", type=str, default=os.path.join("reports", "chapter3_collected_tables"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    export_tables(
        ws_root=os.path.join(PROJECT_ROOT, args.ws_root),
        output_dir=os.path.join(PROJECT_ROOT, args.output_dir),
    )
