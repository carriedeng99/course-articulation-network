"""
match_ipeds.py
==============
Match scraped institution names from panel.csv to IPEDS unit IDs
using the IPEDS HD (Header Data) file (hd2024.csv or similar).

Matching strategy (in order of precedence):
1. Exact name match (case-insensitive)
2. Fuzzy name match (RapidFuzz token_sort_ratio ≥ 85)
3. Geo-tiebreak: among fuzzy ties, prefer institutions in the same state
4. Active-tiebreak: among remaining ties, prefer currently active institutions

Outputs:
  matches.csv  — all scraped names with their best IPEDS match and score
  unmatched.csv — names that fell below the fuzzy threshold (manual review needed)

The resulting matches.csv is used by nj_network.py (via RESOLVED_PATH for
sub-schools like Rutgers that appear under multiple scraped names).

Usage:
  python match_ipeds.py --panel panel.csv --hd hd2024.csv --state NJ --out ./
"""

import argparse
import re
import pandas as pd
from rapidfuzz import fuzz, process

# ── CLI ──────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--panel",  required=True, help="Raw panel CSV from scraper")
parser.add_argument("--hd",     required=True, help="IPEDS HD file (hd2024.csv)")
parser.add_argument("--state",  default=None,  help="Two-letter state code to bias geo-tiebreak (e.g. NJ)")
parser.add_argument("--out",    default=".",   help="Output directory")
parser.add_argument("--threshold", type=float, default=85.0,
                    help="Minimum fuzzy score to accept a match (default: 85)")
args = parser.parse_args()

# ── Load ─────────────────────────────────────────────────────────────────────

panel = pd.read_csv(args.panel, dtype=str)
hd    = pd.read_csv(args.hd, dtype=str, encoding="latin-1")

# Keep relevant HD columns
hd = hd[["UNITID", "INSTNM", "STABBR", "CYACTIVE"]].copy()
hd["INSTNM_lower"] = hd["INSTNM"].str.strip().str.lower()

# Collect unique institution names from both sides of panel
scraped_names = pd.concat([
    panel["sending_institution"],
    panel["receiving_institution"],
]).dropna().str.strip().unique()

print(f"Unique scraped institution names: {len(scraped_names)}")
print(f"IPEDS institutions in HD file   : {len(hd)}")

# ── Exact match ───────────────────────────────────────────────────────────────

exact_map = hd.set_index("INSTNM_lower")[["UNITID", "INSTNM", "STABBR", "CYACTIVE"]].to_dict("index")

records = []

for name in scraped_names:
    name_lower = name.strip().lower()

    # 1. Exact match
    if name_lower in exact_map:
        row = exact_map[name_lower]
        records.append({
            "scraped_name":  name,
            "matched_instnm": row["INSTNM"],
            "unitid":         row["UNITID"],
            "score":          100.0,
            "match_type":     "exact",
            "state_match_rate": None,
        })
        continue

    # 2. Fuzzy match
    choices = hd["INSTNM_lower"].tolist()
    results = process.extract(name_lower, choices, scorer=fuzz.token_sort_ratio,
                              limit=5, score_cutoff=args.threshold)

    if not results:
        records.append({
            "scraped_name":  name,
            "matched_instnm": None,
            "unitid":         None,
            "score":          None,
            "match_type":     "unmatched",
            "state_match_rate": None,
        })
        continue

    # Build candidate frame
    cands = []
    for match_name, score, idx in results:
        hrow = hd.iloc[idx]
        cands.append({
            "instnm":   hrow["INSTNM"],
            "unitid":   hrow["UNITID"],
            "stabbr":   hrow["STABBR"],
            "cyactive": hrow["CYACTIVE"],
            "score":    score,
        })
    cands_df = pd.DataFrame(cands)

    best_score = cands_df["score"].max()
    top = cands_df[cands_df["score"] == best_score]

    match_type = "fuzzy"

    # 3. Geo-tiebreak
    if args.state and len(top) > 1:
        in_state = top[top["stabbr"] == args.state.upper()]
        if not in_state.empty:
            top = in_state
            match_type = "geo"

    # 4. Active-tiebreak
    if len(top) > 1:
        active = top[top["cyactive"] == "1"]
        if not active.empty:
            top = active
            match_type = match_type + "_active_tiebreak" if "geo" in match_type else "fuzzy_active_tiebreak"

    best = top.iloc[0]
    records.append({
        "scraped_name":  name,
        "matched_instnm": best["instnm"],
        "unitid":         best["unitid"],
        "score":          best_score,
        "match_type":     match_type,
        "state_match_rate": None,
    })

# ── Save ─────────────────────────────────────────────────────────────────────

import os
results_df  = pd.DataFrame(records)
matched     = results_df[results_df["match_type"] != "unmatched"]
unmatched   = results_df[results_df["match_type"] == "unmatched"]

out_matches   = os.path.join(args.out, "matches.csv")
out_unmatched = os.path.join(args.out, "unmatched.csv")

results_df.to_csv(out_matches, index=False)
unmatched.to_csv(out_unmatched, index=False)

print(f"\nMatched   : {len(matched)} / {len(results_df)}")
print(f"Unmatched : {len(unmatched)} (saved to {out_unmatched} for manual review)")
print(f"Saved     : {out_matches}")
