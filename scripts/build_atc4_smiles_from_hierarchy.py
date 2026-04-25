#!/usr/bin/env python3
"""
Build ATC-L4 -> SMILES mappings for the current med vocab.

Data flow:
  med vocab code (from voc_final.pkl)
    -> ingredient names from atc_hierarchy.csv
    -> DrugBank names
    -> molecule SMILES

Outputs:
  - atc4toSMILES.pkl: dict[ATC_L4_Code, list[SMILES]]
  - full_smiles_vocab.pkl: sorted unique full-molecule SMILES
  - substructure_smiles_atc4.pkl: sorted unique BRICS fragment SMILES
  - atc4_smiles_report.json: match diagnostics
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import dill
from rdkit import Chem
from rdkit.Chem import BRICS


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Ensure dill can resolve the class path stored inside voc_final.pkl.
from src.util import Voc  # noqa: F401


ATC_HIERARCHY_CSV = REPO_ROOT.parent / "drug-agent" / "data" / "atc_hierarchy.csv"
DRUGBANK_CSV = REPO_ROOT / "data" / "input" / "drugbank_drugs_info.csv"
VOC_PKL = REPO_ROOT / "data" / "output" / "voc_final.pkl"
ATC4_TO_SMILES_PKL = REPO_ROOT / "data" / "atc4toSMILES.pkl"
FULL_SMILES_VOCAB_PKL = REPO_ROOT / "data" / "output" / "full_smiles_vocab.pkl"
SUBSTRUCTURE_SMILES_PKL = REPO_ROOT / "data" / "output" / "substructure_smiles_atc4.pkl"
REPORT_JSON = REPO_ROOT / "data" / "output" / "atc4_smiles_report.json"


NAME_ALIASES = {
    "paracetamol": ["acetaminophen"],
    "glyceryl trinitrate": ["nitroglycerin"],
    "adrenaline": ["epinephrine"],
    "noradrenaline": ["norepinephrine"],
    "salbutamol": ["albuterol"],
    "frusemide": ["furosemide"],
    "lignocaine": ["lidocaine"],
    "phytonadione": ["phytonadione", "phytomenadione"],
    "phytomenadione": ["phytonadione", "phytomenadione"],
    "liquid paraffin": ["mineral oil", "liquid paraffin"],
}


def normalize_name(text: str) -> str:
    text = (text or "").strip().lower()
    text = text.replace("&", " and ")
    text = text.replace("/", " ")
    text = re.sub(r"\([^)]*\)", " ", text)
    text = re.sub(r"[^a-z0-9,\-+ ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def candidate_names(raw_name: str) -> list[str]:
    raw_norm = normalize_name(raw_name)
    candidates = {raw_norm}

    if "," in raw_norm:
        candidates.add(raw_norm.split(",", 1)[0].strip())

    # Remove common descriptive suffixes.
    for suffix in (
        " combinations",
        " combinations excl psycholeptics",
        " combinations with psycholeptics",
        " combinations excl. psycholeptics",
        " combinations with psycholeptics",
        " and diuretics",
        " plain",
    ):
        if raw_norm.endswith(suffix):
            candidates.add(raw_norm[: -len(suffix)].strip())

    expanded = set(candidates)
    for name in list(candidates):
        for alias in NAME_ALIASES.get(name, []):
            expanded.add(normalize_name(alias))

    return [name for name in expanded if name]


def load_med_vocab_codes(voc_path: Path) -> list[str]:
    print(f"[1/5] loading med vocab from {voc_path}", flush=True)
    voc = dill.load(open(voc_path, "rb"))
    med_voc = voc["med_voc"]
    codes = [med_voc.idx2word[i] for i in sorted(med_voc.idx2word)]
    print(f"loaded {len(codes)} med codes", flush=True)
    return codes


def load_atc_l4_to_names(hierarchy_csv: Path, keep_codes: set[str]) -> dict[str, list[str]]:
    print(f"[2/5] loading ATC hierarchy from {hierarchy_csv}", flush=True)
    atc4_to_names = defaultdict(list)
    with hierarchy_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            code = (row.get("ATC_L4_Code") or "").strip()
            name = (row.get("ATC_Name") or "").strip()
            if code in keep_codes and name:
                atc4_to_names[code].append(name)

    deduped = {}
    for code, names in atc4_to_names.items():
        seen = set()
        ordered = []
        for name in names:
            if name not in seen:
                seen.add(name)
                ordered.append(name)
        deduped[code] = ordered
    print(f"loaded names for {len(deduped)} / {len(keep_codes)} med codes", flush=True)
    return deduped


def load_drugbank_name_to_smiles(drugbank_csv: Path) -> dict[str, set[str]]:
    print(f"[3/5] scanning DrugBank rows from {drugbank_csv}", flush=True)
    name_to_smiles = defaultdict(set)
    with drugbank_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for idx, row in enumerate(reader, start=1):
            name = normalize_name(row.get("name") or "")
            smiles = (row.get("moldb_smiles") or "").strip()
            if name and smiles:
                name_to_smiles[name].add(smiles)
            if idx % 50000 == 0:
                print(
                    f"  processed {idx} DrugBank rows, unique names with smiles={len(name_to_smiles)}",
                    flush=True,
                )
    print(f"finished DrugBank scan: {len(name_to_smiles)} unique names with smiles", flush=True)
    return name_to_smiles


def build_atc4_to_smiles(
    med_codes: list[str],
    atc4_to_names: dict[str, list[str]],
    drugbank_name_to_smiles: dict[str, set[str]],
) -> tuple[dict[str, list[str]], dict[str, dict[str, object]]]:
    print("[4/5] matching ATC-L4 codes to DrugBank smiles", flush=True)
    atc4_to_smiles = {}
    report = {}

    for idx, code in enumerate(med_codes, start=1):
        ingredient_names = atc4_to_names.get(code, [])
        matched_smiles = set()
        name_matches = {}
        unmatched_names = []

        for ingredient_name in ingredient_names:
            candidates = candidate_names(ingredient_name)
            hits = set()
            for candidate in candidates:
                hits.update(drugbank_name_to_smiles.get(candidate, set()))
            if hits:
                matched_smiles.update(hits)
                name_matches[ingredient_name] = sorted(hits)
            else:
                unmatched_names.append(ingredient_name)

        if matched_smiles:
            atc4_to_smiles[code] = sorted(matched_smiles)

        report[code] = {
            "ingredient_names": ingredient_names,
            "matched_name_count": len(name_matches),
            "matched_names": name_matches,
            "unmatched_names": unmatched_names,
            "smiles_count": len(matched_smiles),
        }
        if idx % 50 == 0:
            print(
                f"  matched {idx}/{len(med_codes)} codes, codes_with_smiles={len(atc4_to_smiles)}",
                flush=True,
            )

    return atc4_to_smiles, report


def extract_substructure_vocab(atc4_to_smiles: dict[str, list[str]]) -> list[str]:
    print("[5/5] extracting BRICS fragments from matched smiles", flush=True)
    fragments = set()
    total_codes = len(atc4_to_smiles)
    for idx, smiles_list in enumerate(atc4_to_smiles.values(), start=1):
        for smiles in smiles_list:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    continue
                fragments.update(BRICS.BRICSDecompose(mol))
            except Exception:
                continue
        if idx % 50 == 0:
            print(
                f"  processed fragments for {idx}/{total_codes} matched codes, unique_fragments={len(fragments)}",
                flush=True,
            )
    return sorted(fragments)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--atc_hierarchy_csv", type=Path, default=ATC_HIERARCHY_CSV)
    parser.add_argument("--drugbank_csv", type=Path, default=DRUGBANK_CSV)
    parser.add_argument("--voc_pkl", type=Path, default=VOC_PKL)
    parser.add_argument("--atc4_to_smiles_out", type=Path, default=ATC4_TO_SMILES_PKL)
    parser.add_argument("--full_smiles_vocab_out", type=Path, default=FULL_SMILES_VOCAB_PKL)
    parser.add_argument("--substructure_smiles_out", type=Path, default=SUBSTRUCTURE_SMILES_PKL)
    parser.add_argument("--report_json_out", type=Path, default=REPORT_JSON)
    args = parser.parse_args()

    med_codes = load_med_vocab_codes(args.voc_pkl)
    med_code_set = set(med_codes)
    atc4_to_names = load_atc_l4_to_names(args.atc_hierarchy_csv, med_code_set)
    drugbank_name_to_smiles = load_drugbank_name_to_smiles(args.drugbank_csv)
    atc4_to_smiles, report = build_atc4_to_smiles(med_codes, atc4_to_names, drugbank_name_to_smiles)

    full_smiles_vocab = sorted({smiles for smiles_list in atc4_to_smiles.values() for smiles in smiles_list})
    substructure_vocab = extract_substructure_vocab(atc4_to_smiles)

    for out_path in (
        args.atc4_to_smiles_out,
        args.full_smiles_vocab_out,
        args.substructure_smiles_out,
        args.report_json_out,
    ):
        out_path.parent.mkdir(parents=True, exist_ok=True)

    dill.dump(atc4_to_smiles, open(args.atc4_to_smiles_out, "wb"))
    dill.dump(full_smiles_vocab, open(args.full_smiles_vocab_out, "wb"))
    dill.dump(substructure_vocab, open(args.substructure_smiles_out, "wb"))

    matched_codes = sorted(atc4_to_smiles.keys())
    report_summary = {
        "med_vocab_size": len(med_codes),
        "codes_with_names": sum(1 for code in med_codes if atc4_to_names.get(code)),
        "codes_with_smiles": len(matched_codes),
        "coverage_ratio": len(matched_codes) / len(med_codes) if med_codes else 0.0,
        "full_smiles_vocab_size": len(full_smiles_vocab),
        "substructure_vocab_size": len(substructure_vocab),
        "matched_codes": matched_codes,
        "missing_codes": [code for code in med_codes if code not in atc4_to_smiles],
        "details": report,
    }
    with args.report_json_out.open("w", encoding="utf-8") as handle:
        json.dump(report_summary, handle, ensure_ascii=True, indent=2)

    print(f"med vocab size: {len(med_codes)}")
    print(f"codes with hierarchy names: {report_summary['codes_with_names']}")
    print(f"codes with smiles: {len(matched_codes)}")
    print(f"coverage ratio: {report_summary['coverage_ratio']:.2%}")
    print(f"full smiles vocab size: {len(full_smiles_vocab)}")
    print(f"substructure vocab size: {len(substructure_vocab)}")
    print(f"atc4_to_smiles_out: {args.atc4_to_smiles_out}")
    print(f"full_smiles_vocab_out: {args.full_smiles_vocab_out}")
    print(f"substructure_smiles_out: {args.substructure_smiles_out}")
    print(f"report_json_out: {args.report_json_out}")
    print("sample matched codes:", matched_codes[:20])
    print("sample missing codes:", report_summary["missing_codes"][:20])


if __name__ == "__main__":
    main()
