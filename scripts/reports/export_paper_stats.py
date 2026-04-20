import ast
import argparse
import csv
import json
import os
from typing import Dict, List, Optional, Tuple


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATASET_NAME = "assist2009"
DEFAULT_WS_ROOT = os.path.join(PROJECT_ROOT, "ws", "chapter3_runs")
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "reports", "paper_stats")

EXPERIMENT_NAME_TO_VARIANT = {
    "full_model": "Full Model",
    "wo_attention": "w/o attention",
    "wo_query": "w/o query",
    "wo_consistency": "w/o consistency",
}


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


def write_json(path: str, payload) -> None:
    with open(path, "w", encoding="utf8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


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
    max_length = 0
    total_students = 0
    for split in ["train_0.json", "val_0.json", "test.json"]:
        path = os.path.join(PROJECT_ROOT, "data", DATASET_NAME, split)
        data = read_json(path)
        total_students += len(data)
        for stu in data:
            log_len = int(stu[1])
            interactions += log_len
            max_length = max(max_length, log_len)
            for i in range(log_len):
                correct += int(stu[2][i][2])

    avg_length = interactions / total_students if total_students else "N/A"
    return {
        "Dataset": DATASET_NAME,
        "#Students": int(cfg["student_n"]),
        "#Exercises": int(cfg["exer_n"]),
        "#Concepts": int(cfg["knowledge_n"]),
        "#Interactions": int(interactions),
        "Correct Rate": correct / interactions if interactions else "N/A",
        "Avg Length": avg_length,
        "Max Length": int(max_length),
    }


def collect_experiment_results(ws_root: str) -> Dict[str, Dict[str, Dict]]:
    results: Dict[str, Dict[str, Dict]] = {}
    if not os.path.exists(ws_root):
        return results

    for exp_name in sorted(os.listdir(ws_root)):
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
            final_summaries = [r for r in records if r.get("dtype") == "final_summary"]
            latest_final_summary = final_summaries[-1] if final_summaries else None
            stages = [1] if attr_dir_name in {"DIRT_1", "DIRT_2"} else [1, 2]
            stage_rows: Dict[str, Dict] = {}
            for stage in stages:
                best_val = best_validation_record(records, stage)
                if best_val is None:
                    continue
                epoch = int(best_val.get("epoch", -1))
                test_row = test_record_for_epoch(records, stage, epoch)
                stage_rows[f"stage{stage}"] = {
                    "Validation AUC": float(best_val.get("auc", 0.0)),
                    "Validation ACC": float(best_val.get("acc", 0.0)),
                    "Validation RMSE": float(best_val.get("rmse", 0.0)),
                    "Best Epoch": epoch,
                    "Test AUC": float(test_row.get("auc", 0.0)) if test_row else "N/A",
                    "Test ACC": float(test_row.get("acc", 0.0)) if test_row else "N/A",
                    "Test RMSE": float(test_row.get("rmse", 0.0)) if test_row else "N/A",
                }

            final_stage = None
            final_row = {}
            if latest_final_summary is not None:
                summary_stage = int(latest_final_summary.get("best_stage", latest_final_summary.get("stage", -1)))
                final_stage = f"stage{summary_stage}"
                final_row = {
                    "Test AUC": float(latest_final_summary.get("test_auc", 0.0)),
                    "Test ACC": float(latest_final_summary.get("test_acc", 0.0)),
                    "Test RMSE": float(latest_final_summary.get("test_rmse", 0.0)),
                    "Validation AUC": float(latest_final_summary.get("validation_auc", 0.0)),
                    "Validation ACC": float(latest_final_summary.get("validation_acc", 0.0)),
                    "Validation RMSE": float(latest_final_summary.get("validation_rmse", 0.0)),
                    "Best Epoch": int(latest_final_summary.get("best_epoch", -1)),
                }
            attr_map[attr_dir_name] = {
                "config": cfg,
                "stages": stage_rows,
                "final_stage": final_stage,
                "final": final_row,
                "has_final_summary": latest_final_summary is not None,
            }

        if attr_map:
            results[exp_name] = attr_map
    return results


def parse_markdown_tables(md_path: str) -> Dict[str, List[Dict]]:
    tables: Dict[str, List[Dict]] = {}
    if not md_path or not os.path.exists(md_path):
        return tables

    with open(md_path, "r", encoding="utf8") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("**表") and i + 3 < len(lines):
            title = line.strip("*").strip()
            j = i + 1
            while j < len(lines) and not lines[j].strip().startswith("|"):
                j += 1
            if j + 2 >= len(lines):
                i += 1
                continue
            header_line = lines[j].strip()
            separator_line = lines[j + 1].strip()
            if not separator_line.startswith("|"):
                i += 1
                continue
            headers = [item.strip() for item in header_line.strip("|").split("|")]
            rows: List[Dict] = []
            k = j + 2
            while k < len(lines):
                row_line = lines[k].strip()
                if not row_line.startswith("|"):
                    break
                values = [item.strip() for item in row_line.strip("|").split("|")]
                if len(values) == len(headers):
                    row = {headers[idx]: values[idx] for idx in range(len(headers))}
                    rows.append(row)
                k += 1
            if rows:
                tables[title] = rows
            i = k
            continue
        i += 1
    return tables


def build_dataset_table() -> List[Dict]:
    return [dataset_stats_row()]


def build_training_config_table(results: Dict[str, Dict[str, Dict]]) -> List[Dict]:
    cfg = {}
    full = results.get("full_model", {})
    for attr_name in ["DIRT_4", "DIRT_3", "DIRT_2", "DIRT_1"]:
        if attr_name in full:
            cfg = full[attr_name].get("config", {})
            break

    dynamic_weight_value = cfg.get("loss_weight_mode", "N/A")
    if dynamic_weight_value == "plain":
        dynamic_weight_value = "No"

    return [
        {"Setting": "Batch size", "Value": cfg.get("batch_size", "N/A")},
        {
            "Setting": "Epochs (Stage1 / Stage2)",
            "Value": f"{cfg.get('stage1_epochs', 'N/A')} / {cfg.get('stage2_epochs', 'N/A')}",
        },
        {"Setting": "Learning rate", "Value": f"Stage1 = {cfg.get('stage1_lr', 'N/A')}, Stage2 = {cfg.get('stage2_lr', 'N/A')}"},
        {"Setting": "Optimizer", "Value": "Adam"},
        {"Setting": "Seed", "Value": cfg.get("seed", "N/A")},
        {"Setting": "Attention", "Value": "Yes" if int(cfg.get("use_temporal_self_attention", 1)) == 1 else "No"},
        {"Setting": "Query-guided aggregation", "Value": "Yes" if int(cfg.get("use_query_guided_attention", 1)) == 1 else "No"},
        {"Setting": "Dynamic weight", "Value": dynamic_weight_value},
        {"Setting": "Consistency", "Value": "Yes" if int(cfg.get("use_confidence_consistency", 0)) == 1 else "No"},
    ]


def build_main_results_table(results: Dict[str, Dict[str, Dict]], manuscript_tables: Dict[str, List[Dict]]) -> List[Dict]:
    rows: List[Dict] = []
    table_title = "表 3-3 ASSIST2009 上的主结果比较"
    manuscript_rows = manuscript_tables.get(table_title, [])
    seen_models = set()

    for row in manuscript_rows:
        model = row.get("Model")
        if not model:
            continue
        # Prefer manuscript baseline rows, but refresh DIRT+ rows from current logs.
        if model.startswith("DIRT+_"):
            continue
        rows.append({
            "Model": model,
            "AUC": row.get("AUC", "N/A"),
            "ACC": row.get("ACC", "N/A"),
        })
        seen_models.add(model)

    full = results.get("full_model", {})
    for attr_name in ["DIRT_1", "DIRT_2", "DIRT_3", "DIRT_4"]:
        if attr_name not in full:
            continue
        if not full[attr_name].get("has_final_summary"):
            continue
        final = full[attr_name]["final"]
        rows.append({
            "Model": f"DIRT+_{attr_name.split('_')[1]}",
            "AUC": final.get("Test AUC", "N/A"),
            "ACC": final.get("Test ACC", "N/A"),
        })
        seen_models.add(f"DIRT+_{attr_name.split('_')[1]}")

    if not rows:
        for attr_name in ["DIRT_1", "DIRT_2", "DIRT_3", "DIRT_4"]:
            if attr_name not in full:
                continue
            if not full[attr_name].get("has_final_summary"):
                continue
            final = full[attr_name]["final"]
            rows.append({
                "Model": f"DIRT+_{attr_name.split('_')[1]}",
                "AUC": final.get("Test AUC", "N/A"),
                "ACC": final.get("Test ACC", "N/A"),
            })
    return rows


def build_ablation_table(results: Dict[str, Dict[str, Dict]]) -> List[Dict]:
    rows = []
    for exp_name, variant_name in EXPERIMENT_NAME_TO_VARIANT.items():
        exp = results.get(exp_name, {})
        final_rows = []
        for attr_name in ["DIRT_3", "DIRT_4"]:
            if (
                attr_name in exp and
                exp[attr_name].get("has_final_summary") and
                exp[attr_name]["final"]
            ):
                final_rows.append(exp[attr_name]["final"])
        if not final_rows:
            rows.append({"Variant": variant_name, "AUC": "N/A", "ACC": "N/A", "RMSE": "N/A"})
            continue
        agg = aggregate_mean(final_rows, ["Test AUC", "Test ACC", "Test RMSE"])
        rows.append({
            "Variant": variant_name,
            "AUC": agg["Test AUC"],
            "ACC": agg["Test ACC"],
            "RMSE": agg["Test RMSE"],
        })
    return rows


def build_stage_comparison_table(results: Dict[str, Dict[str, Dict]]) -> List[Dict]:
    rows = []
    full = results.get("full_model", {})
    for attr_name in ["DIRT_3", "DIRT_4"]:
        attr = full.get(attr_name, {})
        if not attr.get("has_final_summary"):
            continue
        for stage_name in ["stage1", "stage2"]:
            stage_row = attr.get("stages", {}).get(stage_name, {})
            rows.append({
                "Model": f"DIRT+_{attr_name.split('_')[1]}",
                "Stage": stage_name.capitalize(),
                "Validation AUC": stage_row.get("Validation AUC", "N/A"),
                "Validation ACC": stage_row.get("Validation ACC", "N/A"),
                "Validation RMSE": stage_row.get("Validation RMSE", "N/A"),
                "Test AUC": stage_row.get("Test AUC", "N/A"),
                "Test ACC": stage_row.get("Test ACC", "N/A"),
                "Test RMSE": stage_row.get("Test RMSE", "N/A"),
                "Best Epoch": stage_row.get("Best Epoch", "N/A"),
            })
    return rows


def build_run_metadata(results: Dict[str, Dict[str, Dict]], manuscript_md: str, ws_root: str) -> Dict:
    return {
        "workspace_root": ws_root,
        "dataset": DATASET_NAME,
        "experiments_found": sorted(results.keys()),
        "manuscript_md": manuscript_md if manuscript_md else "",
        "generated_files": [
            "table_3_1_dataset_statistics.csv",
            "table_3_2_training_config.csv",
            "table_3_3_main_results.csv",
            "table_3_4_ablation.csv",
            "table_3_5_stage_comparison.csv",
            "paper_stats.md",
            "paper_stats.json",
            "README.md",
        ],
    }


def write_readme(output_dir: str) -> None:
    readme_path = os.path.join(output_dir, "README.md")
    content = (
        "# Paper Stats\n\n"
        "This folder stores paper-ready data exports for the DIRT+ manuscript.\n\n"
        "Files:\n"
        "- `table_3_1_dataset_statistics.csv`: dataset statistics used in the paper.\n"
        "- `table_3_2_training_config.csv`: current training configuration from `full_model`.\n"
        "- `table_3_3_main_results.csv`: main results table. Baseline rows can come from the manuscript, DIRT+ rows come from current logs.\n"
        "- `table_3_4_ablation.csv`: ablation results aggregated from `DIRT_3` and `DIRT_4` test metrics.\n"
        "- `table_3_5_stage_comparison.csv`: stage1/stage2 comparison for `DIRT+_3` and `DIRT+_4`, with validation and test metrics kept in separate columns.\n"
        "- `paper_stats.md`: markdown preview of all exported tables.\n"
        "- `paper_stats.json`: machine-readable bundle of all exported tables.\n"
    )
    with open(readme_path, "w", encoding="utf8") as f:
        f.write(content)


def export_tables(ws_root: str, output_dir: str, manuscript_md: str) -> None:
    ensure_dir(output_dir)
    results = collect_experiment_results(ws_root)
    manuscript_tables = parse_markdown_tables(manuscript_md)

    dataset_rows = build_dataset_table()
    training_rows = build_training_config_table(results)
    main_rows = build_main_results_table(results, manuscript_tables)
    ablation_rows = build_ablation_table(results)
    stage_rows = build_stage_comparison_table(results)

    tables = {
        "table_3_1_dataset_statistics": dataset_rows,
        "table_3_2_training_config": training_rows,
        "table_3_3_main_results": main_rows,
        "table_3_4_ablation": ablation_rows,
        "table_3_5_stage_comparison": stage_rows,
    }

    for name, rows in tables.items():
        fieldnames = list(rows[0].keys()) if rows else []
        write_csv(os.path.join(output_dir, f"{name}.csv"), fieldnames, rows)

    md_sections = ["# Paper Stats", ""]
    for name, rows in tables.items():
        fieldnames = list(rows[0].keys()) if rows else []
        md_sections.append(f"## {name}")
        md_sections.append("")
        md_sections.append(markdown_table(fieldnames, rows))
        md_sections.append("")

    with open(os.path.join(output_dir, "paper_stats.md"), "w", encoding="utf8") as f:
        f.write("\n".join(md_sections))

    write_json(
        os.path.join(output_dir, "paper_stats.json"),
        {
            "metadata": build_run_metadata(results, manuscript_md, ws_root),
            "tables": tables,
        },
    )
    write_readme(output_dir)


def parse_args():
    parser = argparse.ArgumentParser(description="Export paper-ready DIRT+ statistics into a dedicated folder.")
    parser.add_argument("--ws_root", type=str, default=DEFAULT_WS_ROOT)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--manuscript_md", type=str, default="")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    export_tables(
        ws_root=args.ws_root,
        output_dir=args.output_dir,
        manuscript_md=args.manuscript_md,
    )
