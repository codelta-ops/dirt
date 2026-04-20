import argparse
import os
import sys
from typing import Dict, List


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "reports", "chapter3_final_tables")
DEFAULT_WS_ROOT = os.path.join(PROJECT_ROOT, "ws", "chapter3_runs")

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.reports.export_paper_stats import (  # noqa: E402
    build_ablation_table,
    build_dataset_table,
    build_main_results_table,
    build_stage_comparison_table,
    build_training_config_table,
    collect_experiment_results,
    ensure_dir,
    markdown_table,
    parse_markdown_tables,
    write_csv,
)


def export_all(ws_root: str, output_dir: str, manuscript_md: str) -> None:
    ensure_dir(output_dir)
    results = collect_experiment_results(ws_root)
    manuscript_tables = parse_markdown_tables(manuscript_md)

    tables: Dict[str, List[Dict]] = {
        "table1_main_results": build_main_results_table(results, manuscript_tables),
        "table2_ablation": build_ablation_table(results),
        "table3_stage_comparison": build_stage_comparison_table(results),
        "table4_dataset_statistics": build_dataset_table(),
        "table5_training_config": build_training_config_table(results),
    }

    md_lines: List[str] = []
    for title, rows in tables.items():
        fieldnames = list(rows[0].keys()) if rows else []
        write_csv(os.path.join(output_dir, f"{title}.csv"), fieldnames, rows)
        md_lines.append(f"## {title}")
        md_lines.append("")
        md_lines.append(markdown_table(fieldnames, rows))
        md_lines.append("")

    with open(os.path.join(output_dir, "chapter3_final_tables.md"), "w", encoding="utf8") as f:
        f.write("\n".join(md_lines))


def parse_args():
    parser = argparse.ArgumentParser(description="Export final Chapter 3 tables from current experiment logs.")
    parser.add_argument("--ws_root", type=str, default=DEFAULT_WS_ROOT)
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--manuscript_md", type=str, default="")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    export_all(
        ws_root=args.ws_root,
        output_dir=args.output_dir,
        manuscript_md=args.manuscript_md,
    )
