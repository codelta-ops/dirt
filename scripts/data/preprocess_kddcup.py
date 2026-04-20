import argparse
import csv
import json
import os
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Sequence, Tuple


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_INPUT_TRAIN = os.path.join(PROJECT_ROOT, "data", "kddcup", "bridge_to_algebra_2006_2007_train.txt")
DEFAULT_INPUT_TEST = os.path.join(PROJECT_ROOT, "data", "kddcup", "bridge_to_algebra_2006_2007_test.txt")
DEFAULT_INPUT_TEST_LABELS = os.path.join(PROJECT_ROOT, "data", "kddcup", "bridge_to_algebra_2006_2007.txt")
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "kddcup")


@dataclass
class RawRecord:
    order_id: str
    user_id: str
    item_key: str
    correct: int
    skill_names: Tuple[str, ...]
    timestamp: datetime


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess KDDCup Bridge to Algebra into the project JSON format.")
    parser.add_argument("--input_train", type=str, default=DEFAULT_INPUT_TRAIN)
    parser.add_argument("--input_test", type=str, default=DEFAULT_INPUT_TEST)
    parser.add_argument("--input_test_labels", type=str, default=DEFAULT_INPUT_TEST_LABELS)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min_log", type=int, default=15)
    parser.add_argument("--max_log", type=int, default=200)
    parser.add_argument("--test_ratio", type=float, default=0.2)
    parser.add_argument("--folds", type=int, default=5)
    return parser.parse_args()


def parse_timestamp(row: Dict[str, str]) -> datetime:
    for key in ["First Transaction Time", "Step Start Time", "Correct Transaction Time", "Step End Time"]:
        raw_value = (row.get(key) or "").strip()
        if not raw_value:
            continue
        for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
            try:
                return datetime.strptime(raw_value, fmt)
            except ValueError:
                continue
    raise ValueError("No valid timestamp field found.")


def parse_skill_names(raw_kc: str) -> Tuple[str, ...]:
    return tuple(item.strip() for item in raw_kc.split("~~") if item.strip())


def build_item_key(row: Dict[str, str]) -> str:
    problem_name = (row.get("Problem Name") or "").strip()
    step_name = (row.get("Step Name") or "").strip()
    if step_name:
        return f"{problem_name}::{step_name}"
    return problem_name


def read_test_labels(path: str) -> Dict[str, str]:
    labels: Dict[str, str] = {}
    with open(path, "r", encoding="utf8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            labels[(row.get("Row") or "").strip()] = (row.get("Correct First Attempt") or "").strip()
    return labels


def read_valid_records(input_train: str, input_test: str, input_test_labels: str) -> Tuple[List[RawRecord], Dict[str, int]]:
    csv.field_size_limit(1024 * 1024 * 16)
    stats = {
        "raw_rows": 0,
        "train_rows": 0,
        "test_rows": 0,
        "dropped_missing_user": 0,
        "dropped_missing_item": 0,
        "dropped_missing_kc": 0,
        "dropped_invalid_correct": 0,
        "dropped_invalid_timestamp": 0,
    }
    test_labels = read_test_labels(input_test_labels)
    records: List[RawRecord] = []

    for source_name, path in [("train", input_train), ("test", input_test)]:
        with open(path, "r", encoding="utf8", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                stats["raw_rows"] += 1
                stats[f"{source_name}_rows"] += 1

                user_id = (row.get("Anon Student Id") or "").strip()
                item_key = build_item_key(row)
                raw_kc = (row.get("KC(SubSkills)") or "").strip()
                correct_raw = (row.get("Correct First Attempt") or "").strip()
                order_id = (row.get("Row") or "").strip()
                if source_name == "test":
                    correct_raw = test_labels.get(order_id, "")

                if not user_id:
                    stats["dropped_missing_user"] += 1
                    continue
                if not item_key:
                    stats["dropped_missing_item"] += 1
                    continue
                if not raw_kc:
                    stats["dropped_missing_kc"] += 1
                    continue
                if correct_raw not in {"0", "1"}:
                    stats["dropped_invalid_correct"] += 1
                    continue

                skill_names = parse_skill_names(raw_kc)
                if not skill_names:
                    stats["dropped_missing_kc"] += 1
                    continue

                try:
                    timestamp = parse_timestamp(row)
                except ValueError:
                    stats["dropped_invalid_timestamp"] += 1
                    continue

                records.append(
                    RawRecord(
                        order_id=order_id or f"{user_id}_{stats['raw_rows']}",
                        user_id=user_id,
                        item_key=item_key,
                        correct=int(correct_raw),
                        skill_names=skill_names,
                        timestamp=timestamp,
                    )
                )
    return records, stats


def filter_low_frequency_items(records: Sequence[RawRecord], min_log: int) -> Tuple[List[RawRecord], int]:
    item_counts = Counter(record.item_key for record in records)
    kept = [record for record in records if item_counts[record.item_key] >= min_log]
    return kept, len(records) - len(kept)


def build_sequences(records: Sequence[RawRecord], max_log: int, min_log: int) -> Tuple[List[List[RawRecord]], Dict[str, int]]:
    stats = {
        "students_before_segmentation": 0,
        "students_after_segmentation": 0,
        "dropped_short_sequences": 0,
    }
    grouped: Dict[str, List[RawRecord]] = defaultdict(list)
    for record in records:
        grouped[record.user_id].append(record)

    stats["students_before_segmentation"] = len(grouped)
    sequences: List[List[RawRecord]] = []
    for user_records in grouped.values():
        user_records = sorted(user_records, key=lambda item: (item.timestamp, item.order_id))
        for start in range(0, len(user_records), max_log):
            chunk = user_records[start:start + max_log]
            if len(chunk) < min_log:
                stats["dropped_short_sequences"] += 1
                continue
            sequences.append(chunk)

    stats["students_after_segmentation"] = len(sequences)
    return sequences, stats


def split_train_test(
    sequences: Sequence[List[RawRecord]],
    test_ratio: float,
    seed: int,
) -> Tuple[List[List[RawRecord]], List[List[RawRecord]]]:
    shuffled = list(sequences)
    random.Random(seed).shuffle(shuffled)
    if len(shuffled) <= 1:
        return shuffled, []
    test_count = max(1, int(round(len(shuffled) * test_ratio)))
    test_count = min(test_count, len(shuffled) - 1)
    return shuffled[:-test_count], shuffled[-test_count:]


def build_maps(
    sequences: Sequence[List[RawRecord]],
) -> Tuple[Dict[str, int], Dict[str, int], Dict[Tuple[int, ...], int]]:
    item_keys = sorted({record.item_key for sequence in sequences for record in sequence})
    skill_names = sorted({skill for sequence in sequences for record in sequence for skill in record.skill_names})
    item_map = {item_key: idx + 1 for idx, item_key in enumerate(item_keys)}
    skill_map = {skill_name: idx + 1 for idx, skill_name in enumerate(skill_names)}
    skill_comb_map: Dict[Tuple[int, ...], int] = {}
    for sequence in sequences:
        for record in sequence:
            skill_tuple = tuple(sorted(skill_map[skill] for skill in record.skill_names))
            if skill_tuple not in skill_comb_map:
                skill_comb_map[skill_tuple] = len(skill_comb_map) + 1
    return item_map, skill_map, skill_comb_map


def convert_sequences(
    sequences: Sequence[List[RawRecord]],
    item_map: Dict[str, int],
    skill_map: Dict[str, int],
    skill_comb_map: Dict[Tuple[int, ...], int],
) -> List[List]:
    converted = []
    for idx, sequence in enumerate(sequences, start=1):
        logs = []
        for record in sequence:
            skill_ids = sorted(skill_map[skill] for skill in record.skill_names)
            skill_tuple = tuple(skill_ids)
            logs.append([
                record.order_id,
                item_map[record.item_key],
                record.correct,
                skill_ids,
                skill_comb_map[skill_tuple],
            ])
        converted.append([idx, len(logs), logs])
    return converted


def write_json(path: str, data) -> None:
    with open(path, "w", encoding="utf8") as f:
        json.dump(data, f, ensure_ascii=False)


def write_data_config(
    output_dir: str,
    student_n: int,
    exer_n: int,
    knowledge_n: int,
    skill_comb_n: int,
    min_log: int,
    max_log: int,
    stu_cnt_train: int,
) -> None:
    config = {
        "student_n": student_n,
        "exer_n": exer_n,
        "knowledge_n": knowledge_n,
        "skill_comb_n": skill_comb_n,
        "min_log": min_log,
        "max_log": max_log,
        "stu_cnt_train": stu_cnt_train,
    }
    with open(os.path.join(output_dir, "data_config.txt"), "w", encoding="utf8") as f:
        f.write(str(config))


def write_fold_splits(output_dir: str, train_sequences: Sequence[List], folds: int, seed: int) -> Dict[str, int]:
    shuffled = list(train_sequences)
    random.Random(seed).shuffle(shuffled)
    for fold_idx in range(folds):
        val_split = [item for idx, item in enumerate(shuffled) if idx % folds == fold_idx]
        train_split = [item for idx, item in enumerate(shuffled) if idx % folds != fold_idx]
        write_json(os.path.join(output_dir, f"train_{fold_idx}.json"), train_split)
        write_json(os.path.join(output_dir, f"val_{fold_idx}.json"), val_split)
    return {
        "folds": folds,
        "train_0_size": len([item for idx, item in enumerate(shuffled) if idx % folds != 0]),
        "val_0_size": len([item for idx, item in enumerate(shuffled) if idx % folds == 0]),
    }


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    records, read_stats = read_valid_records(args.input_train, args.input_test, args.input_test_labels)
    records, dropped_low_freq = filter_low_frequency_items(records, args.min_log)
    sequences, seq_stats = build_sequences(records, args.max_log, args.min_log)
    train_raw, test_raw = split_train_test(sequences, args.test_ratio, args.seed)

    all_sequences = list(train_raw) + list(test_raw)
    item_map, skill_map, skill_comb_map = build_maps(all_sequences)
    train_sequences = convert_sequences(train_raw, item_map, skill_map, skill_comb_map)
    test_sequences = convert_sequences(test_raw, item_map, skill_map, skill_comb_map)

    fold_stats = write_fold_splits(args.output_dir, train_sequences, args.folds, args.seed)
    write_json(os.path.join(args.output_dir, "test.json"), test_sequences)
    write_data_config(
        output_dir=args.output_dir,
        student_n=len(all_sequences),
        exer_n=len(item_map),
        knowledge_n=len(skill_map),
        skill_comb_n=len(skill_comb_map),
        min_log=args.min_log,
        max_log=args.max_log,
        stu_cnt_train=len(train_sequences),
    )

    stat_payload = {
        "dataset": "kddcup",
        "source_train": os.path.abspath(args.input_train),
        "source_test": os.path.abspath(args.input_test),
        "source_test_labels": os.path.abspath(args.input_test_labels),
        "seed": args.seed,
        "min_log": args.min_log,
        "max_log": args.max_log,
        "test_ratio": args.test_ratio,
        "folds": args.folds,
        "paper_rule_notes": [
            "Merged original KDDCup train and test rows, and filled test labels from the label file.",
            "Dropped rows without KC annotation.",
            "Dropped items with fewer than 15 responses.",
            "Sorted responses by student and timestamp.",
            "Split sequences longer than 200 into contiguous dummy-student chunks.",
            "Dropped sequences shorter than 15 after segmentation.",
            "Used an 80/20 random split over preprocessed response sequences.",
            "Created 5-fold train/validation files from the training portion as a project convention.",
        ],
        "read_stats": read_stats,
        "dropped_low_frequency_item_rows": dropped_low_freq,
        "sequence_stats": seq_stats,
        "train_sequence_count": len(train_sequences),
        "test_sequence_count": len(test_sequences),
        "student_n": len(all_sequences),
        "exer_n": len(item_map),
        "knowledge_n": len(skill_map),
        "skill_comb_n": len(skill_comb_map),
        "fold_stats": fold_stats,
    }
    write_json(os.path.join(args.output_dir, "stat.json"), stat_payload)

    print(
        "[done]",
        f"train_sequences={len(train_sequences)}",
        f"test_sequences={len(test_sequences)}",
        f"students={len(all_sequences)}",
        f"exercises={len(item_map)}",
        f"skills={len(skill_map)}",
    )


if __name__ == "__main__":
    main()
