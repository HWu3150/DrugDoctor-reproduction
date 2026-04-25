#!/usr/bin/env python3
"""
Build a new substructure_smiles vocab from an existing ATC4->SMILES mapping.

Only ATC4 codes present in the current med vocab are considered. Codes without
SMILES coverage are skipped and treated as having no structure prior.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import dill
from rdkit import Chem
from rdkit.Chem import BRICS


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.util import Voc  # noqa: F401


def load_pickle(path: Path):
    with path.open("rb") as handle:
        try:
            return pickle.load(handle)
        except Exception:
            handle.seek(0)
            return dill.load(handle)


def med_codes_from_vocab(voc_path: Path) -> list[str]:
    voc = dill.load(open(voc_path, "rb"))
    med_voc = voc["med_voc"]
    return [med_voc.idx2word[i] for i in sorted(med_voc.idx2word)]


def unique_fragments_from_smiles(smiles_by_code: dict[str, list[str]]) -> list[str]:
    fragments = set()
    total_codes = len(smiles_by_code)
    for idx, (code, smiles_list) in enumerate(sorted(smiles_by_code.items()), start=1):
        for smiles in smiles_list:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    continue
                fragments.update(BRICS.BRICSDecompose(mol))
            except Exception:
                continue
        if idx % 25 == 0:
            print(
                f"  processed {idx}/{total_codes} matched codes, unique_fragments={len(fragments)}",
                flush=True,
            )
    return sorted(fragments)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--voc_pkl", type=Path, default=Path("data/output/voc_final.pkl"))
    parser.add_argument("--atc4_smiles_pkl", type=Path, default=Path("data/atc4toSMILES.pkl"))
    parser.add_argument(
        "--matched_smiles_out",
        type=Path,
        default=Path("data/output/atc4toSMILES_covered.pkl"),
        help="Filtered ATC4->SMILES mapping restricted to current med vocab.",
    )
    parser.add_argument(
        "--substructure_out",
        type=Path,
        default=Path("data/output/substructure_smiles_atc4.pkl"),
        help="Output pkl containing sorted unique BRICS fragment SMILES.",
    )
    parser.add_argument(
        "--report_out",
        type=Path,
        default=Path("data/output/substructure_smiles_atc4_report.json"),
        help="Output report json for coverage debugging.",
    )
    args = parser.parse_args()

    print(f"[1/3] loading med vocab from {args.voc_pkl}", flush=True)
    med_codes = med_codes_from_vocab(args.voc_pkl)
    med_set = set(med_codes)
    print(f"loaded {len(med_codes)} med vocab codes", flush=True)

    print(f"[2/3] loading ATC4->SMILES from {args.atc4_smiles_pkl}", flush=True)
    atc4_to_smiles = load_pickle(args.atc4_smiles_pkl)
    covered = {
        str(code).strip(): list(smiles_list)
        for code, smiles_list in atc4_to_smiles.items()
        if str(code).strip() in med_set
    }
    matched_codes = sorted(covered)
    missing_codes = sorted(med_set - set(matched_codes))
    print(
        f"matched {len(matched_codes)}/{len(med_codes)} med codes "
        f"({len(matched_codes)/len(med_codes):.2%})",
        flush=True,
    )

    print("[3/3] extracting BRICS fragments", flush=True)
    substructures = unique_fragments_from_smiles(covered)

    args.matched_smiles_out.parent.mkdir(parents=True, exist_ok=True)
    args.substructure_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.parent.mkdir(parents=True, exist_ok=True)

    dill.dump(covered, open(args.matched_smiles_out, "wb"))
    dill.dump(substructures, open(args.substructure_out, "wb"))

    report = {
        "med_vocab_size": len(med_codes),
        "matched_code_count": len(matched_codes),
        "coverage_ratio": len(matched_codes) / len(med_codes) if med_codes else 0.0,
        "missing_code_count": len(missing_codes),
        "substructure_count": len(substructures),
        "matched_codes": matched_codes,
        "missing_codes": missing_codes,
    }
    with args.report_out.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=True, indent=2)

    print(f"substructure count: {len(substructures)}", flush=True)
    print(f"matched_smiles_out: {args.matched_smiles_out}", flush=True)
    print(f"substructure_out: {args.substructure_out}", flush=True)
    print(f"report_out: {args.report_out}", flush=True)
    print(f"missing sample: {missing_codes[:20]}", flush=True)


if __name__ == "__main__":
    main()
