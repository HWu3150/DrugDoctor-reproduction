#!/usr/bin/env python3
"""
Inspect which drugs are covered by DrugDoctor's chemistry/DDI resources and
optionally compare that coverage against an external vocabulary CSV.

Notes
-----
- `substructure_smiles.pkl` itself stores fragment SMILES, not drug codes.
  For drug-level chemistry coverage, this script uses `atc3toSMILES.pkl`.
- `ddi_A_final.pkl` is only an adjacency matrix. To recover which drug code each
  row/column refers to, you also need `voc_final.pkl` so the script can read
  `med_voc.idx2word`.

Examples
--------
python scripts/check_drug_coverage.py \
  --atc3-smiles-pkl data/output/atc3toSMILES.pkl \
  --ddi-pkl data/output/ddi_A_final.pkl \
  --voc-pkl data/output/voc_final.pkl \
  --vocab-csv /home/hanwei/projects/drug-agent/data/atc_hierarchy.csv \
  --vocab-column ATC_L4_Code
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from pathlib import Path
from typing import Any, Iterable

def _load_pickle(path: Path) -> Any:
    try:
        with path.open("rb") as handle:
            return pickle.load(handle)
    except ModuleNotFoundError as exc:
        if exc.name != "dill":
            raise
    except Exception:
        pass

    try:
        import dill  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"Failed to load {path}. This file likely needs dill; install it first."
        ) from exc

    with path.open("rb") as handle:
        return dill.load(handle)


def _sorted_codes(codes: Iterable[str]) -> list[str]:
    return sorted({str(code).strip() for code in codes if str(code).strip()})


def _prefix_codes(codes: Iterable[str], length: int) -> list[str]:
    return _sorted_codes(code[:length] for code in codes)


def _read_vocab_csv(path: Path, column: str) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if column not in (reader.fieldnames or []):
            raise ValueError(
                f"Column {column!r} not found in {path}. Available columns: {reader.fieldnames}"
            )
        return _sorted_codes(row[column] for row in reader if row.get(column))


def _get_med_codes_from_voc(voc_obj: Any) -> list[str]:
    if not isinstance(voc_obj, dict) or "med_voc" not in voc_obj:
        raise ValueError("voc_final.pkl does not contain a 'med_voc' entry")
    med_voc = voc_obj["med_voc"]
    idx2word = getattr(med_voc, "idx2word", None)
    if not isinstance(idx2word, dict):
        raise ValueError("med_voc.idx2word is missing or malformed")
    return [str(idx2word[i]).strip() for i in sorted(idx2word.keys())]


def _summarize_overlap(name: str, external: set[str], covered: set[str]) -> dict[str, Any]:
    overlap = sorted(external & covered)
    missing = sorted(external - covered)
    return {
        "name": name,
        "external_count": len(external),
        "covered_count": len(covered),
        "overlap_count": len(overlap),
        "missing_count": len(missing),
        "coverage_ratio": (len(overlap) / len(external)) if external else 0.0,
        "overlap_codes": overlap,
        "missing_codes": missing,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--substructure-smiles-pkl",
        type=Path,
        default=Path("data/output/substructure_smiles.pkl"),
        help="Path to substructure_smiles.pkl (fragment SMILES only; used for sanity stats).",
    )
    parser.add_argument(
        "--atc3-smiles-pkl",
        type=Path,
        default=Path("data/output/atc3toSMILES.pkl"),
        help="Path to atc3toSMILES.pkl. Its keys are DrugDoctor's chemistry-covered drugs.",
    )
    parser.add_argument(
        "--ddi-pkl",
        type=Path,
        default=Path("data/output/ddi_A_final.pkl"),
        help="Path to ddi_A_final.pkl.",
    )
    parser.add_argument(
        "--voc-pkl",
        type=Path,
        default=Path("data/output/voc_final.pkl"),
        help="Path to voc_final.pkl. Required to map ddi rows/cols back to drug codes.",
    )
    parser.add_argument(
        "--vocab-csv",
        type=Path,
        default=None,
        help="Optional external vocab CSV to compare against, e.g. drug-agent/data/atc_hierarchy.csv.",
    )
    parser.add_argument(
        "--vocab-column",
        default="ATC_L4_Code",
        help="Column in --vocab-csv to use as the external vocabulary.",
    )
    parser.add_argument(
        "--prefix-len",
        type=int,
        default=4,
        help="Prefix length used to compare a finer-grained external vocab against DrugDoctor's 4-char codes.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory for JSON/TXT reports.",
    )
    args = parser.parse_args()

    report: dict[str, Any] = {"inputs": {k: str(v) for k, v in vars(args).items()}}

    substructures = _load_pickle(args.substructure_smiles_pkl)
    if not isinstance(substructures, list):
        raise ValueError(f"{args.substructure_smiles_pkl} is not a list")
    report["substructure_smiles"] = {
        "fragment_count": len(substructures),
        "sample_fragments": substructures[:10],
    }

    atc3_to_smiles = _load_pickle(args.atc3_smiles_pkl)
    if not isinstance(atc3_to_smiles, dict):
        raise ValueError(f"{args.atc3_smiles_pkl} is not a dict")
    chemistry_drugs = _sorted_codes(atc3_to_smiles.keys())
    report["chemistry_drugs"] = {
        "count": len(chemistry_drugs),
        "sample_codes": chemistry_drugs[:20],
        "all_codes": chemistry_drugs,
    }

    ddi_codes: list[str] | None = None
    ddi_nonzero_codes: list[str] | None = None
    if args.voc_pkl.exists():
        try:
            import numpy as np  # type: ignore
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Parsing ddi_A_final.pkl requires numpy; install it first."
            ) from exc

        ddi_adj = _load_pickle(args.ddi_pkl)
        voc = _load_pickle(args.voc_pkl)
        ddi_codes = _sorted_codes(_get_med_codes_from_voc(voc))

        ddi_np = np.asarray(ddi_adj)
        if ddi_np.ndim != 2 or ddi_np.shape[0] != ddi_np.shape[1]:
            raise ValueError(f"{args.ddi_pkl} is not a square matrix")
        if len(ddi_codes) != ddi_np.shape[0]:
            raise ValueError(
                f"DDI matrix size {ddi_np.shape} does not match med vocab size {len(ddi_codes)}"
            )

        nonzero_idx = np.where(ddi_np.sum(axis=1) > 0)[0].tolist()
        ddi_nonzero_codes = [ddi_codes[i] for i in nonzero_idx]
        report["ddi_drugs"] = {
            "matrix_shape": list(ddi_np.shape),
            "all_vocab_drug_count": len(ddi_codes),
            "nonzero_ddi_drug_count": len(ddi_nonzero_codes),
            "sample_all_vocab_codes": ddi_codes[:20],
            "sample_nonzero_ddi_codes": ddi_nonzero_codes[:20],
            "all_vocab_codes": ddi_codes,
            "nonzero_ddi_codes": ddi_nonzero_codes,
        }
    else:
        report["ddi_drugs"] = {
            "warning": f"{args.voc_pkl} does not exist; cannot map ddi rows/cols back to drug codes"
        }

    if args.vocab_csv:
        external_codes = _read_vocab_csv(args.vocab_csv, args.vocab_column)
        external_prefix = _prefix_codes(external_codes, args.prefix_len)
        report["external_vocab"] = {
            "count": len(external_codes),
            "prefix_count": len(external_prefix),
            "sample_codes": external_codes[:20],
            "sample_prefix_codes": external_prefix[:20],
        }

        chemistry_set = set(chemistry_drugs)
        report["comparisons"] = {
            "chemistry_exact": _summarize_overlap(
                "chemistry_exact", set(external_codes), chemistry_set
            ),
            "chemistry_prefix": _summarize_overlap(
                "chemistry_prefix", set(external_prefix), chemistry_set
            ),
        }

        if ddi_codes is not None and ddi_nonzero_codes is not None:
            ddi_all_set = set(ddi_codes)
            ddi_nonzero_set = set(ddi_nonzero_codes)
            report["comparisons"]["ddi_vocab_exact"] = _summarize_overlap(
                "ddi_vocab_exact", set(external_codes), ddi_all_set
            )
            report["comparisons"]["ddi_vocab_prefix"] = _summarize_overlap(
                "ddi_vocab_prefix", set(external_prefix), ddi_all_set
            )
            report["comparisons"]["ddi_nonzero_exact"] = _summarize_overlap(
                "ddi_nonzero_exact", set(external_codes), ddi_nonzero_set
            )
            report["comparisons"]["ddi_nonzero_prefix"] = _summarize_overlap(
                "ddi_nonzero_prefix", set(external_prefix), ddi_nonzero_set
            )

    print("=== DrugDoctor Resource Summary ===")
    print(f"substructure fragments: {report['substructure_smiles']['fragment_count']}")
    print(f"chemistry-covered drugs (from atc3toSMILES): {len(chemistry_drugs)}")
    ddi_info = report["ddi_drugs"]
    if "warning" in ddi_info:
        print(f"ddi drug mapping: {ddi_info['warning']}")
    else:
        print(
            "ddi vocab drugs: "
            f"{ddi_info['all_vocab_drug_count']} total, "
            f"{ddi_info['nonzero_ddi_drug_count']} with at least one ddi edge"
        )

    if "comparisons" in report:
        print("\n=== Coverage Against External Vocab ===")
        for key, summary in report["comparisons"].items():
            print(
                f"{key}: overlap={summary['overlap_count']}/{summary['external_count']} "
                f"({summary['coverage_ratio']:.2%}), missing={summary['missing_count']}"
            )

    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        json_path = args.output_dir / "drug_coverage_report.json"
        with json_path.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=True, indent=2)

        chemistry_path = args.output_dir / "chemistry_drugs.txt"
        chemistry_path.write_text("\n".join(chemistry_drugs) + "\n", encoding="utf-8")

        if ddi_codes is not None:
            ddi_all_path = args.output_dir / "ddi_vocab_drugs.txt"
            ddi_all_path.write_text("\n".join(ddi_codes) + "\n", encoding="utf-8")
        if ddi_nonzero_codes is not None:
            ddi_nonzero_path = args.output_dir / "ddi_nonzero_drugs.txt"
            ddi_nonzero_path.write_text(
                "\n".join(ddi_nonzero_codes) + "\n", encoding="utf-8"
            )

        print(f"\nWrote report to {json_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
