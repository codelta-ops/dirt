import argparse
import csv
import json
import math
import os
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Sequence, Tuple


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_INPUT_CSV = os.path.join(PROJECT_ROOT, "data", "assist2012", "2012-2013-data-with-predictions-4-final.csv")
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "assist2012")


@dataclass
class RawRecord:
    order_id: str
    user_id: str
    problem_id: str
    correct: int
    skill_id: str
    timestamp: datetime


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess ASSIST2012 into the project JSON format.")
    parser.add_argument("--input_csv", type=str, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min_log", type=int, default=15)
    parser.add_argument("--max_log", type=int, default=200)
    parser.add_argument("--test_ratio", type=float, default=0.2)
    parser.add_argument("--folds", type=int, default=5)
    return parser.parse_args()


def parse_timestamp(raw_value: str) -> datetime:
    raw_value = raw_value.strip()
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(raw_value, fmt)
        except ValueError:
            continue
    raise ValueError(f"Unsupported timestamp format: {raw_value}")


def read_valid_records(input_csv: str) -> Tuple[List[RawRecord], Dict[str, int]]:
    csv.field_size_limit(1024 * 1024 * 64)
    stats = {
        "raw_rows": 0,
        "dropped_missing_problem": 0,
        "dropped_missing_user": 0,
        "dropped_missing_skill": 0,
        "dropped_invalid_correct": 0,
        "dropped_invalid_timestamp": 0,
    }
    records: List[RawRecord] = []
    with open(input_csv, "r", encoding="utf8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            stats["raw_rows"] += 1

            user_id = row.get("user_id", "").strip()
            problem_id = row.get("problem_id", "").strip()
            skill_id = row.get("skill_id", "").strip()
            correct_raw = row.get("correct", "").strip()
            order_id = (row.get("problem_log_id") or row.get("problemlogid") or "").strip()
            start_time = row.get("start_time", "").strip()

            if not problem_id:
                stats["dropped_missing_problem"] += 1
                continue
            if not user_id:
                stats["dropped_missing_user"] += 1
                continue
            if not skill_id:
                stats["dropped_missing_skill"] += 1
                continue
            if correct_raw not in {"0", "1"}:
                stats["dropped_invalid_correct"] += 1
                continue
            try:
                timestamp = parse_timestamp(start_time)
            except ValueError:
                stats["dropped_invalid_timestamp"] += 1
                continue

            records.append(
                RawRecord(
                    order_id=order_id or f"{user_id}_{problem_id}_{stats['raw_rows']}",
                    user_id=user_id,
                    problem_id=problem_id,
                    correct=int(correct_raw),
                    skill_id=skill_id,
                    timestamp=timestamp,
                )
            )
    return records, stats


def filter_low_frequency_problems(records: Sequence[RawRecord], min_log: int) -> Tuple[List[RawRecord], int]:
    problem_counts = Counter(record.problem_id for record in records)
    kept_records = [record for record in records if problem_counts[record.problem_id] >= min_log]
    dropped = len(records) - len(kept_records)
    return kept_records, dropped


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
    train_count = len(shuffled) - test_count
    return shuffled[:train_count], shuffled[train_count:]


def build_id_maps(sequences: Sequence[List[RawRecord]]) -> Tuple[Dict[str, int], Dict[str, int], Dict[str, int], Dict[Tuple[int, ...], int]]:
    problem_ids = sorted({record.problem_id for sequence in sequences for record in sequence}, key=int)
    skill_ids = sorted({record.skill_id for sequence in sequences for record in sequence}, key=int)
    exercise_map = {problem_id: idx + 1 for idx, problem_id in enumerate(problem_ids)}
    skill_map = {skill_id: idx + 1 for idx, skill_id in enumerate(skill_ids)}
    skill_comb_map: Dict[Tuple[int, ...], int] = {}

    for sequence in sequences:
        for record in sequence:
            skill_tuple = (skill_map[record.skill_id],)
            if skill_tuple not in skill_comb_map:
                skill_comb_map[skill_tuple] = len(skill_comb_map) + 1

    student_map = {f"segment_{idx}": idx + 1 for idx in range(len(sequences))}
    return student_map, exercise_map, skill_map, skill_comb_map


def convert_sequences_to_json(
    sequences: Sequence[List[RawRecord]],
    exercise_map: Dict[str, int],
    skill_map: Dict[str, int],
    skill_comb_map: Dict[Tuple[int, ...], int],
) -> List[List]:
    converted: List[List] = []
    for idx, sequence in enumerate(sequences, start=1):
        logs = []
        for record in sequence:
            skill_id = skill_map[record.skill_id]
            skill_tuple = (skill_id,)
            logs.append([
                record.order_id,
                exercise_map[record.problem_id],
                record.correct,
                [skill_id],
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

    records, read_stats = read_valid_records(args.input_csv)
    records, dropped_low_freq = filter_low_frequency_problems(records, args.min_log)
    sequences, seq_stats = build_sequences(records, args.max_log, args.min_log)
    train_sequences_raw, test_sequences_raw = split_train_test(sequences, args.test_ratio, args.seed)

    all_sequences = list(train_sequences_raw) + list(test_sequences_raw)
    _, exercise_map, skill_map, skill_comb_map = build_id_maps(all_sequences)
    train_sequences = convert_sequences_to_json(train_sequences_raw, exercise_map, skill_map, skill_comb_map)
    test_sequences = convert_sequences_to_json(test_sequences_raw, exercise_map, skill_map, skill_comb_map)

    fold_stats = write_fold_splits(args.output_dir, train_sequences, args.folds, args.seed)
    write_json(os.path.join(args.output_dir, "test.json"), test_sequences)

    write_data_config(
        output_dir=args.output_dir,
        student_n=len(all_sequences),
        exer_n=len(exercise_map),
        knowledge_n=len(skill_map),
        skill_comb_n=len(skill_comb_map),
        min_log=args.min_log,
        max_log=args.max_log,
        stu_cnt_train=len(train_sequences),
    )

    stat_payload = {
        "dataset": "assist2012",
        "source_csv": os.path.abspath(args.input_csv),
        "seed": args.seed,
        "min_log": args.min_log,
        "max_log": args.max_log,
        "test_ratio": args.test_ratio,
        "folds": args.folds,
        "paper_rule_notes": [
            "Dropped rows without KC annotation.",
            "Dropped problems with fewer than 15 responses.",
            "Sorted responses by student and timestamp.",
            "Split sequences longer than 200 into contiguous dummy-student chunks.",
            "Dropped sequences shorter than 15 after segmentation.",
            "Used an 80/20 random split over preprocessed response sequences.",
            "Created 5-fold train/validation files from the training portion as a project convention.",
        ],
        "read_stats": read_stats,
        "dropped_low_frequency_problem_rows": dropped_low_freq,
        "sequence_stats": seq_stats,
        "train_sequence_count": len(train_sequences),
        "test_sequence_count": len(test_sequences),
        "student_n": len(all_sequences),
        "exer_n": len(exercise_map),
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
        f"exercises={len(exercise_map)}",
        f"skills={len(skill_map)}",
    )


if __name__ == "__main__":
    main()
