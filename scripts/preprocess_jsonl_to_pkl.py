#!/usr/bin/env python3
import argparse
import os

import dill

from src.util import load_jsonl_data_and_voc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--jsonl_path",
        type=str,
        required=True,
        help="Path to structured jsonl with records.{diagnosis,procedure,medication}.",
    )
    parser.add_argument(
        "--records_out",
        type=str,
        default="data/output/records_final.pkl",
        help="Output path for encoded patient records.",
    )
    parser.add_argument(
        "--voc_out",
        type=str,
        default="data/output/voc_final.pkl",
        help="Output path for vocab dictionary.",
    )
    parser.add_argument(
        "--preview_patients",
        type=int,
        default=1,
        help="How many encoded patients to preview after preprocessing.",
    )
    parser.add_argument(
        "--preview_visits",
        type=int,
        default=2,
        help="How many visits to preview for each preview patient.",
    )
    parser.add_argument(
        "--preview_vocab",
        type=int,
        default=10,
        help="How many vocab entries to print for each code space.",
    )
    args = parser.parse_args()

    records, voc = load_jsonl_data_and_voc(args.jsonl_path)

    records_dir = os.path.dirname(args.records_out)
    voc_dir = os.path.dirname(args.voc_out)
    if records_dir:
        os.makedirs(records_dir, exist_ok=True)
    if voc_dir:
        os.makedirs(voc_dir, exist_ok=True)

    dill.dump(records, open(args.records_out, "wb"))
    dill.dump(voc, open(args.voc_out, "wb"))

    diag_size = len(voc["diag_voc"].idx2word)
    proc_size = len(voc["pro_voc"].idx2word)
    med_size = len(voc["med_voc"].idx2word)
    total_visits = sum(len(patient) for patient in records)

    print(f"patients: {len(records)}")
    print(f"visits: {total_visits}")
    print(f"diag vocab size: {diag_size}")
    print(f"proc vocab size: {proc_size}")
    print(f"med vocab size: {med_size}")
    print(f"records_out: {args.records_out}")
    print(f"voc_out: {args.voc_out}")

    preview_vocab = max(args.preview_vocab, 0)
    if preview_vocab:
        print("\n=== vocab preview ===")
        print(
            "diag idx2word:",
            [(idx, voc["diag_voc"].idx2word[idx]) for idx in range(min(preview_vocab, diag_size))],
        )
        print(
            "proc idx2word:",
            [(idx, voc["pro_voc"].idx2word[idx]) for idx in range(min(preview_vocab, proc_size))],
        )
        print(
            "med idx2word:",
            [(idx, voc["med_voc"].idx2word[idx]) for idx in range(min(preview_vocab, med_size))],
        )

    preview_patients = min(max(args.preview_patients, 0), len(records))
    preview_visits = max(args.preview_visits, 0)
    if preview_patients and preview_visits:
        print("\n=== encoded records preview ===")
        for patient_idx in range(preview_patients):
            patient = records[patient_idx]
            print(f"patient[{patient_idx}] visits={len(patient)}")
            for visit_idx, visit in enumerate(patient[:preview_visits]):
                diag_ids, proc_ids, med_ids = visit
                print(
                    f"  visit[{visit_idx}] "
                    f"diag_ids={diag_ids} proc_ids={proc_ids} med_ids={med_ids}"
                )


if __name__ == "__main__":
    main()
