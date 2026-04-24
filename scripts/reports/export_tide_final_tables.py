import argparse
import ast
import csv
import json
import os
from typing import Dict, Iterable, List, Optional, Tuple


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "reports", "tide_final_tables")
DEFAULT_FULL_WS_ROOT = os.path.join(PROJECT_ROOT, "ws", "chapter3_runs")
DEFAULT_BASE_WS_ROOT = os.path.join(PROJECT_ROOT, "ws", "dirt")
DEFAULT_DATASETS = ["assist2009", "assist2012", "kddcup"]

DATASET_DISPLAY = {
    "assist2009": "ASSIST2009",
    "assist2012": "ASSIST2012",
    "kddcup": "KDDCup",
}

OFFICIAL_TIDE_CONFIG = {
    "cross_idx": 0,
    "seed": 2024,
    "batch_size": 32,
    "stage1_epochs": 10,
    "stage2_epochs": 5,
    "stage1_lr": 0.002,
    "stage2_lr": 0.002,
    "optimizer": "Adam",
    "attention": "Yes",
    "query_guided_aggregation": "Yes",
    "dynamic_weight": "learnable",
    "consistency": "Yes",
    "scheduler": "ReduceLROnPlateau (monitor validation AUC, patience=1)",
    "stage2_scope": "TIDE_3 / TIDE_4 only",
}

MAIN_TABLE_HEADERS = [
    "Model",
    "ASSIST2009 AUC",
    "ASSIST2009 ACC",
    "ASSIST2012 AUC",
    "ASSIST2012 ACC",
    "KDDCup AUC",
    "KDDCup ACC",
]

ABLATION_SUMMARY_HEADERS = ["Variant", "Dataset", "Seed", "AUC", "ACC", "RMSE"]
ABLATION_DETAIL_HEADERS = ["Variant", "Model", "Dataset", "Seed", "AUC", "ACC", "RMSE"]
STAGE_HEADERS = [
    "Dataset",
    "Model",
    "Stage",
    "Validation AUC",
    "Validation ACC",
    "Validation RMSE",
    "Test AUC",
    "Test ACC",
    "Test RMSE",
    "Best Epoch",
]
LONG_HEADERS = ["模型", "数据集", "种子", "AUC", "ACC"]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def read_json(path: str):
    with open(path, "r", encoding="utf8") as f:
        return json.load(f)


def read_jsonl(path: str) -> List[Dict]:
    rows: List[Dict] = []
    if not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_csv(path: str, headers: List[str], rows: List[Dict]) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({header: row.get(header, "") for header in headers})


def write_json(path: str, payload) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def fmt(value) -> str:
    if value in [None, ""]:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def markdown_table(headers: List[str], rows: List[Dict]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(h, "")) for h in headers) + " |")
    return "\n".join(lines)


def dataset_stats_row(dataset: str) -> Dict:
    cfg_path = os.path.join(PROJECT_ROOT, "data", dataset, "data_config.txt")
    if not os.path.exists(cfg_path):
        return {
            "Dataset": DATASET_DISPLAY[dataset],
            "#Students": "",
            "#Exercises": "",
            "#Concepts": "",
            "#Interactions": "",
            "Correct Rate": "",
        }

    with open(cfg_path, "r", encoding="utf8") as f:
        cfg = ast.literal_eval(f.read().strip())

    interactions = 0
    correct = 0
    for split in ["train_0.json", "val_0.json", "test.json"]:
        split_path = os.path.join(PROJECT_ROOT, "data", dataset, split)
        if not os.path.exists(split_path):
            continue
        data = read_json(split_path)
        for stu in data:
            log_len = int(stu[1])
            interactions += log_len
            for i in range(log_len):
                correct += int(stu[2][i][2])

    return {
        "Dataset": DATASET_DISPLAY[dataset],
        "#Students": int(cfg["student_n"]),
        "#Exercises": int(cfg["exer_n"]),
        "#Concepts": int(cfg["knowledge_n"]),
        "#Interactions": int(interactions),
        "Correct Rate": float(correct / interactions) if interactions else "",
    }


def collect_dataset_table(datasets: Iterable[str]) -> List[Dict]:
    return [dataset_stats_row(dataset) for dataset in datasets]


def collect_training_config_table() -> List[Dict]:
    cfg = OFFICIAL_TIDE_CONFIG
    return [
        {"Setting": "cross_idx", "Value": cfg["cross_idx"]},
        {"Setting": "seed", "Value": cfg["seed"]},
        {"Setting": "batch size", "Value": cfg["batch_size"]},
        {"Setting": "epochs (stage1 / stage2)", "Value": f"{cfg['stage1_epochs']} / {cfg['stage2_epochs']}"},
        {"Setting": "learning rate", "Value": f"stage1={cfg['stage1_lr']}, stage2={cfg['stage2_lr']}"},
        {"Setting": "optimizer", "Value": cfg["optimizer"]},
        {"Setting": "causal temporal enhancement", "Value": cfg["attention"]},
        {"Setting": "target-aware history aggregation", "Value": cfg["query_guided_aggregation"]},
        {"Setting": "dynamic step weight", "Value": cfg["dynamic_weight"]},
        {"Setting": "stage2 consistency", "Value": cfg["consistency"]},
        {"Setting": "scheduler", "Value": cfg["scheduler"]},
        {"Setting": "stage2 applies to", "Value": cfg["stage2_scope"]},
    ]


def full_metrics_path(full_ws_root: str, dataset: str, seed: int, exp_name: str, attr_idx: int) -> str:
    return os.path.join(full_ws_root, dataset, f"seed_{seed}", exp_name, f"DIRT_{attr_idx}", "metrics.jsonl")


def base_metrics_path(base_ws_root: str, dataset: str, attr_idx: int) -> str:
    return os.path.join(base_ws_root, dataset, f"DIRT_{attr_idx}", "metrics.jsonl")


def latest_record(records: List[Dict], dtype: str) -> Optional[Dict]:
    candidates = [row for row in records if row.get("dtype") == dtype]
    return candidates[-1] if candidates else None


def validation_priority_key(record: Dict) -> Tuple[float, float, float, int]:
    return (
        float(record.get("auc", 0.0)),
        float(record.get("acc", 0.0)),
        -float(record.get("rmse", float("inf"))),
        -int(record.get("epoch", 0)),
    )


def best_validation_record(records: List[Dict], stage: int) -> Optional[Dict]:
    candidates = [
        row for row in records
        if row.get("dtype") == "validation" and int(row.get("stage", -1)) == int(stage)
    ]
    if not candidates:
        return None
    return max(candidates, key=validation_priority_key)


def matching_test_record(records: List[Dict], stage: int, epoch: int) -> Optional[Dict]:
    for row in records:
        if (
            row.get("dtype") == "test" and
            int(row.get("stage", -1)) == int(stage) and
            int(row.get("epoch", -1)) == int(epoch)
        ):
            return row
    return None


def collect_base_result(base_ws_root: str, dataset: str, attr_idx: int, seed: int) -> Optional[Dict]:
    path = base_metrics_path(base_ws_root, dataset, attr_idx)
    records = read_jsonl(path)
    if not records:
        return None

    test_row = latest_record(records, "test")
    if test_row is None:
        return None

    return {
        "模型": f"DIRT_{attr_idx}",
        "数据集": DATASET_DISPLAY[dataset],
        "种子": seed,
        "AUC": float(test_row.get("auc", 0.0)),
        "ACC": float(test_row.get("acc", 0.0)),
        "RMSE": float(test_row.get("rmse", 0.0)),
        "stage": int(test_row.get("stage", 1)),
        "epoch": int(test_row.get("epoch", -1)),
    }


def collect_full_result(full_ws_root: str, dataset: str, attr_idx: int, seed: int, exp_name: str = "full_model") -> Optional[Dict]:
    path = full_metrics_path(full_ws_root, dataset, seed, exp_name, attr_idx)
    records = read_jsonl(path)
    if not records:
        return None

    final_summary = latest_record(records, "final_summary")
    if final_summary is not None:
        return {
            "模型": f"TIDE_{attr_idx}",
            "数据集": DATASET_DISPLAY[dataset],
            "种子": seed,
            "AUC": float(final_summary.get("test_auc", 0.0)),
            "ACC": float(final_summary.get("test_acc", 0.0)),
            "RMSE": float(final_summary.get("test_rmse", 0.0)),
            "stage": int(final_summary.get("best_stage", final_summary.get("stage", -1))),
            "epoch": int(final_summary.get("best_epoch", -1)),
        }

    test_row = latest_record(records, "test")
    if test_row is None:
        return None
    return {
        "模型": f"TIDE_{attr_idx}",
        "数据集": DATASET_DISPLAY[dataset],
        "种子": seed,
        "AUC": float(test_row.get("auc", 0.0)),
        "ACC": float(test_row.get("acc", 0.0)),
        "RMSE": float(test_row.get("rmse", 0.0)),
        "stage": int(test_row.get("stage", -1)),
        "epoch": int(test_row.get("epoch", -1)),
    }


def collect_long_rows(full_ws_root: str, base_ws_root: str, datasets: Iterable[str], seed: int) -> List[Dict]:
    rows: List[Dict] = []
    for dataset in datasets:
        for attr_idx in [1, 2, 3, 4]:
            base_row = collect_base_result(base_ws_root, dataset, attr_idx, seed)
            if base_row is not None:
                rows.append({header: base_row[header] for header in LONG_HEADERS})
            full_row = collect_full_result(full_ws_root, dataset, attr_idx, seed)
            if full_row is not None:
                rows.append({header: full_row[header] for header in LONG_HEADERS})
    return sorted(rows, key=lambda row: (row["数据集"], row["模型"], row["种子"]))


def collect_main_table(long_rows: List[Dict]) -> List[Dict]:
    model_order = [f"DIRT_{i}" for i in range(1, 5)] + [f"TIDE_{i}" for i in range(1, 5)]
    lookup = {(row["模型"], row["数据集"]): row for row in long_rows}
    rows: List[Dict] = []
    for model in model_order:
        row = {"Model": model}
        for dataset in ["ASSIST2009", "ASSIST2012", "KDDCup"]:
            found = lookup.get((model, dataset), {})
            row[f"{dataset} AUC"] = found.get("AUC", "")
            row[f"{dataset} ACC"] = found.get("ACC", "")
        rows.append(row)
    return rows


def mean_or_blank(values: List[float]):
    return sum(values) / len(values) if values else ""


def collect_ablation_tables(full_ws_root: str, seed: int) -> Tuple[List[Dict], List[Dict]]:
    variants = {
        "Full Model": "full_model",
        "w/o attention": "wo_attention",
        "w/o query": "wo_query",
        "w/o weight": "wo_weight",
        "w/o consistency": "wo_consistency",
    }
    detail_rows: List[Dict] = []
    summary_rows: List[Dict] = []
    dataset = "assist2009"

    for variant_name, exp_name in variants.items():
        variant_rows: List[Dict] = []
        for attr_idx in [1, 2, 3, 4]:
            row = collect_full_result(full_ws_root, dataset, attr_idx, seed, exp_name=exp_name)
            if row is None:
                continue
            detail_row = {
                "Variant": variant_name,
                "Model": row["模型"],
                "Dataset": row["数据集"],
                "Seed": row["种子"],
                "AUC": row["AUC"],
                "ACC": row["ACC"],
                "RMSE": row["RMSE"],
            }
            detail_rows.append(detail_row)
            variant_rows.append(detail_row)

        summary_rows.append({
            "Variant": variant_name,
            "Dataset": DATASET_DISPLAY[dataset],
            "Seed": seed,
            "AUC": mean_or_blank([float(row["AUC"]) for row in variant_rows]),
            "ACC": mean_or_blank([float(row["ACC"]) for row in variant_rows]),
            "RMSE": mean_or_blank([float(row["RMSE"]) for row in variant_rows]),
        })

    return summary_rows, detail_rows


def collect_stage_table(full_ws_root: str, datasets: Iterable[str], seed: int) -> List[Dict]:
    rows: List[Dict] = []
    for dataset in datasets:
        for attr_idx in [3, 4]:
            path = full_metrics_path(full_ws_root, dataset, seed, "full_model", attr_idx)
            records = read_jsonl(path)
            if not records:
                continue
            for stage in [1, 2]:
                best_val = best_validation_record(records, stage)
                if best_val is None:
                    continue
                epoch = int(best_val.get("epoch", -1))
                test_row = matching_test_record(records, stage, epoch)
                rows.append({
                    "Dataset": DATASET_DISPLAY[dataset],
                    "Model": f"TIDE_{attr_idx}",
                    "Stage": f"Stage{stage}",
                    "Validation AUC": float(best_val.get("auc", 0.0)),
                    "Validation ACC": float(best_val.get("acc", 0.0)),
                    "Validation RMSE": float(best_val.get("rmse", 0.0)),
                    "Test AUC": float(test_row.get("auc", 0.0)) if test_row else "",
                    "Test ACC": float(test_row.get("acc", 0.0)) if test_row else "",
                    "Test RMSE": float(test_row.get("rmse", 0.0)) if test_row else "",
                    "Best Epoch": epoch,
                })
    return rows


def write_readme(output_dir: str, metadata: Dict) -> None:
    path = os.path.join(output_dir, "README.md")
    content = [
        "# TIDE Final Tables",
        "",
        "This folder is the single official export location for paper tables and long-format results.",
        "",
        f"- Full result root: `{metadata['full_ws_root']}`",
        f"- Base result root: `{metadata['base_ws_root']}`",
        f"- Datasets: `{', '.join(metadata['datasets'])}`",
        f"- Seed: `{metadata['seed']}`",
        "",
        "Files:",
        "- `table_3_1_dataset_statistics.csv`",
        "- `table_3_2_training_config.csv`",
        "- `table_3_3_main_results.csv`",
        "- `table_3_4_ablation.csv`",
        "- `table_3_4_ablation_detail.csv`",
        "- `table_3_5_stage_comparison.csv`",
        "- `long_results.csv`",
        "- `tables.md`",
        "- `tables.json`",
        "",
        "Notes:",
        "- `table_3_3_main_results.csv` uses only official TIDE and base DIRT result paths.",
        "- `table_3_4_ablation.csv` averages `TIDE_1~4` on ASSIST2009.",
        "- `table_3_5_stage_comparison.csv` reports `TIDE_3 / TIDE_4` stage-wise metrics.",
        "- If workspaces are empty, result tables will be created with headers but no metrics.",
        "",
    ]
    with open(path, "w", encoding="utf8") as f:
        f.write("\n".join(content))


def export_tables(full_ws_root: str, base_ws_root: str, output_dir: str, datasets: Iterable[str], seed: int) -> None:
    ensure_dir(output_dir)

    dataset_rows = collect_dataset_table(datasets)
    training_rows = collect_training_config_table()
    long_rows = collect_long_rows(full_ws_root, base_ws_root, datasets, seed)
    main_rows = collect_main_table(long_rows)
    ablation_rows, ablation_detail_rows = collect_ablation_tables(full_ws_root, seed)
    stage_rows = collect_stage_table(full_ws_root, datasets, seed)

    write_csv(os.path.join(output_dir, "table_3_1_dataset_statistics.csv"), list(dataset_rows[0].keys()), dataset_rows)
    write_csv(os.path.join(output_dir, "table_3_2_training_config.csv"), ["Setting", "Value"], training_rows)
    write_csv(os.path.join(output_dir, "table_3_3_main_results.csv"), MAIN_TABLE_HEADERS, main_rows)
    write_csv(os.path.join(output_dir, "table_3_4_ablation.csv"), ABLATION_SUMMARY_HEADERS, ablation_rows)
    write_csv(os.path.join(output_dir, "table_3_4_ablation_detail.csv"), ABLATION_DETAIL_HEADERS, ablation_detail_rows)
    write_csv(os.path.join(output_dir, "table_3_5_stage_comparison.csv"), STAGE_HEADERS, stage_rows)
    write_csv(os.path.join(output_dir, "long_results.csv"), LONG_HEADERS, long_rows)

    tables_md = [
        "# TIDE Final Tables",
        "",
        "## table_3_1_dataset_statistics",
        "",
        markdown_table(list(dataset_rows[0].keys()), dataset_rows),
        "",
        "## table_3_2_training_config",
        "",
        markdown_table(["Setting", "Value"], training_rows),
        "",
        "## table_3_3_main_results",
        "",
        markdown_table(MAIN_TABLE_HEADERS, main_rows),
        "",
        "## table_3_4_ablation",
        "",
        markdown_table(ABLATION_SUMMARY_HEADERS, ablation_rows),
        "",
        "## table_3_4_ablation_detail",
        "",
        markdown_table(ABLATION_DETAIL_HEADERS, ablation_detail_rows),
        "",
        "## table_3_5_stage_comparison",
        "",
        markdown_table(STAGE_HEADERS, stage_rows),
        "",
        "## long_results",
        "",
        markdown_table(LONG_HEADERS, long_rows),
        "",
    ]
    with open(os.path.join(output_dir, "tables.md"), "w", encoding="utf8") as f:
        f.write("\n".join(tables_md))

    metadata = {
        "full_ws_root": full_ws_root,
        "base_ws_root": base_ws_root,
        "datasets": list(datasets),
        "seed": seed,
        "counts": {
            "dataset_rows": len(dataset_rows),
            "training_rows": len(training_rows),
            "long_rows": len(long_rows),
            "main_rows": len(main_rows),
            "ablation_rows": len(ablation_rows),
            "ablation_detail_rows": len(ablation_detail_rows),
            "stage_rows": len(stage_rows),
        },
    }
    write_json(
        os.path.join(output_dir, "tables.json"),
        {
            "metadata": metadata,
            "table_3_1_dataset_statistics": dataset_rows,
            "table_3_2_training_config": training_rows,
            "table_3_3_main_results": main_rows,
            "table_3_4_ablation": ablation_rows,
            "table_3_4_ablation_detail": ablation_detail_rows,
            "table_3_5_stage_comparison": stage_rows,
            "long_results": long_rows,
        },
    )
    write_readme(output_dir, metadata)


def parse_args():
    parser = argparse.ArgumentParser(description="Unified exporter for official TIDE paper tables.")
    parser.add_argument("--full_ws_root", type=str, default=DEFAULT_FULL_WS_ROOT)
    parser.add_argument("--base_ws_root", type=str, default=DEFAULT_BASE_WS_ROOT)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--datasets", type=str, default="assist2009,assist2012,kddcup")
    parser.add_argument("--seed", type=int, default=OFFICIAL_TIDE_CONFIG["seed"])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    datasets = [item.strip() for item in args.datasets.split(",") if item.strip()]
    export_tables(
        full_ws_root=args.full_ws_root,
        base_ws_root=args.base_ws_root,
        output_dir=args.output_dir,
        datasets=datasets,
        seed=args.seed,
    )
