"""
nj_network.py
=============
Builds a transfer-equivalency network from New Jersey's panel data and
extracts connected components as contrastive-learning groups.

Pipeline
--------
1.  Load raw panel
2.  Merge resolved manual-review UNITIDs (Rutgers sub-schools)
3.  Deduplicate
4.  Normalize subject prefixes
5.  Drop catch-all equivalencies (incl. range-code nodes)
6.  Drop rows still missing IPEDS IDs or subject prefixes
7.  Construct a bipartite course graph
8.  Extract connected components → contrastive learning groups
9.  Score each component with embedding similarity
10. Flag suspect components
11. Save outputs
12. Visualizations
"""

import os
import re
import warnings
import numpy as np
import pandas as pd
import networkx as nx
import scipy.sparse as sp
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from networkx.algorithms.community import greedy_modularity_communities

# ── Sentence-transformer import (graceful fallback to TF-IDF if unavailable) ──
try:
    from sentence_transformers import SentenceTransformer
    USE_SENTENCE_TRANSFORMER = True
except ImportError:
    USE_SENTENCE_TRANSFORMER = False
    print("sentence-transformers not installed — falling back to TF-IDF similarity.")
    print("Install with: pip install sentence-transformers")

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────

RAW_PATH      = "panel.csv"
RESOLVED_PATH = "manual_review_resolved.csv"
OUT_PANEL      = "data/panel_clean.csv"
OUT_COMPONENTS = "data/components.csv"
OUT_REVIEW     = "data/review_flagged.csv"

# ── Thresholds ────────────────────────────────────────────────────────────────

SIZE_THRESHOLD        = 10
HOMOGENEITY_THRESHOLD = 0.8
# Lowered from 0.5 → 0.45 to recover borderline-similar components (45 additional
# components confirmed via analysis, spread across CS, MATH, HIST, CRJU, BIOL).
# TF-IDF fallback uses 0.3 (different score distribution).
SIMILARITY_THRESHOLD  = 0.45 if USE_SENTENCE_TRANSFORMER else 0.3

# ── 1. Load ───────────────────────────────────────────────────────────────────

df       = pd.read_csv(RAW_PATH, dtype=str)
resolved = pd.read_csv(RESOLVED_PATH, dtype=str)

print(f"Raw panel: {len(df)} rows")
print(f"Similarity encoder: {'sentence-transformer (all-MiniLM-L6-v2)' if USE_SENTENCE_TRANSFORMER else 'TF-IDF (fallback)'}")
print(f"Similarity threshold: {SIMILARITY_THRESHOLD}")

# ── 2. Merge resolved UNITIDs ─────────────────────────────────────────────────

nj_resolved = (
    resolved[resolved["State"] == "NJ"]
    .set_index("scraped_name")[["confirmed_unitid"]]
    .rename(columns={"confirmed_unitid": "_resolved_unitid"})
)

df = df.join(nj_resolved, on="receiving_institution", how="left")
filled = df["receiving_institution_ipeds_id"].isna() & df["_resolved_unitid"].notna()
df.loc[filled, "receiving_institution_ipeds_id"] = df.loc[filled, "_resolved_unitid"]
df = df.drop(columns="_resolved_unitid")
print(f"Resolved missing receiving UNITIDs: {filled.sum()} rows")

# ── 3. Deduplicate ────────────────────────────────────────────────────────────

PAIR_KEY = [
    "sending_institution", "sending_sub_pref", "sending_course",
    "sending_course_title", "receiving_institution", "receiving_sub_pref",
    "receiving_course", "receiving_course_title",
]
before = len(df)
df = df.drop_duplicates(subset=PAIR_KEY).reset_index(drop=True)
print(f"Dedup: {before} → {len(df)} rows ({before - len(df)} removed)")

df["state"] = "NJ"

# ── 4. Normalize subject prefixes ─────────────────────────────────────────────
# Audit notes:
# - THE: confirmed genuine Theatre at all NJ institutions (not Theology) — keep
# - CMP: confirmed Computer Science courses — added to CS group
# - DAT: primarily data processing / CS courses — added to CS group
#   (note: some DAT courses are data science / R programming; acceptable under CS)

SUBJ_NORM = {
    # Accounting
    "AC": "ACCT", "ACC": "ACCT", "ACCTX": "ACCT",
    # Biology
    "BI": "BIOL", "BIO": "BIOL", "BIOS": "BIOL",
    # Business / management
    "BU": "BUS", "BUA": "BUS", "BUSA": "BUS", "BUSE": "BUS",
    "BUSN": "BUS", "BUSI": "BUS", "MAN": "MGT", "MGMT": "MGT",
    # Chemistry
    "CH": "CHEM", "CHE": "CHEM", "CHM": "CHEM", "CHEL": "CHEM",
    "CHEMX": "CHEM",
    # Computer science / IT  ← CMP and DAT added
    "CIS": "CS", "CISY": "CS", "CSC": "CS", "CSIT": "CS",
    "CSE": "CS", "CSS": "CS", "COMP": "CS", "CPS": "CS",
    "CMPS": "CS", "CMPT": "CS", "CSCI": "CS",
    "CMP": "CS",   # confirmed CS: C programming, Java, VB.NET, systems analysis
    "DAT": "CS",   # confirmed primarily CS/data-processing; includes some R courses
    # Criminal justice
    "CJ": "CRJU", "CJS": "CRJU", "CRIM": "CRJU", "CJI": "CRJU",
    "CRJS": "CRJU", "CRMJ": "CRJU", "CRJ": "CRJU",
    # Dance
    "DAN": "DANC", "DNC": "DANC", "DNCE": "DANC",
    # Economics
    "EC": "ECON", "ECO": "ECON",
    # Education
    "EDU": "EDUC", "ED": "EDUC",
    # English / writing
    "EN": "ENG", "ENGL": "ENG", "ENGH": "ENG", "WRT": "ENG",
    "WRIT": "ENG", "ENGW": "ENG", "ENWR": "ENG",
    # Finance
    "FI": "FIN", "FINC": "FIN",
    # French
    "FR": "FREN", "FRE": "FREN", "FRN": "FREN",
    # Geography / geology
    "GEO": "GEOG",
    # German
    "GER": "GERM",
    # Health sciences
    "HLT": "HLTH", "HSC": "HLTH", "HLSC": "HLTH", "HES": "HLTH",
    # History
    "HI": "HIST", "HIS": "HIST",
    # Humanities
    "HUM": "HUMN",
    # Italian
    "ITA": "ITAL", "ITL": "ITAL",
    # Japanese
    "JPN": "JAPN",
    # Journalism
    "JRN": "JOUR",
    # Marketing
    "MKT": "MRKT", "MKTG": "MRKT",
    # Mathematics
    "MA": "MATH", "MAT": "MATH", "MTH": "MATH",
    # Music
    "MU": "MUS", "MUSC": "MUS", "MUSI": "MUS", "MUSIC": "MUS",
    # Nursing
    "NUR": "NURS",
    # Philosophy
    "PHI": "PHIL", "PHL": "PHIL",
    # Physical education / kinesiology
    "PE": "PHED", "PED": "PHED",
    # Physics
    "PHY": "PHYS",
    # Political science
    "POL": "POLS", "POLI": "POLS", "PS": "POLS", "PSC": "POLS",
    "POS": "POLS",
    # Psychology
    "PSY": "PSYC", "PY": "PSYC",
    # Sociology
    "SO": "SOC", "SOCI": "SOC", "SOCL": "SOC",
    # Spanish
    "SP": "SPAN", "SPA": "SPAN", "SPN": "SPAN",
    # Speech / communication
    "SPCH": "COMM", "CMN": "COMM", "COM": "COMM", "COMS": "COMM",
    "SPCM": "COMM",
    # Theatre  ← THE audited: confirmed genuine Theatre at all NJ institutions
    "THA": "THEA", "THE": "THEA", "THR": "THEA", "THTR": "THEA",
}

def normalize_subj(s):
    if pd.isna(s):
        return s
    return SUBJ_NORM.get(s.strip().upper(), s.strip().upper())

df["sending_sub_pref"]   = df["sending_sub_pref"].apply(normalize_subj)
df["receiving_sub_pref"] = df["receiving_sub_pref"].apply(normalize_subj)
print(f"Subject prefixes normalized using {len(SUBJ_NORM)}-entry map")

# ── 5. Drop catch-all / non-specific equivalencies ────────────────────────────

CATCHALL_PATTERNS = [
    r"\belective\b",
    r"\bfree elective\b",
    r"\btransfer credit\b",
    r"\btopics in\b",
    r"\bspecial topics\b",
    r"\blower.?division\b",
    r"\bupper.?division\b",
    r"\bgeneral studies\b",
    r"\bopen elective\b",
    r"\bundistributed\b",
    r"\bno equivalent\b",
    r"\bnon.?equivalent\b",
    r"\btba\b",
    r"\bto be arranged\b",
    r"\bno credit\b",
    r"\bno direct equivalent\b",
    r"\btransfer intermediate level\b",
    r"\blower level transfer\b",
    r"\bcourse area credit\b",
    r"\baged courses\b",
    r"\bremedial.?developmental\b",
    r"\b\d+\s+level\s+\w+\s+elec",
]

# Range-code patterns: course numbers that are placeholders rather than real courses
# e.g. 1XX, 2XX (multi-digit + double X) and 1X, 2X (single-digit + single X)
RANGE_CODE_PATTERNS = [
    r"^\d+XX$",   # e.g. 1XX, 2XX, 100XX
    r"^\d+X$",    # e.g. 1X, 2X, 10X  ← added to catch single-X placeholders
]

def is_catchall(title):
    if pd.isna(title):
        return False
    t = title.lower()
    return any(re.search(p, t) for p in CATCHALL_PATTERNS)

def is_range_code(course):
    if pd.isna(course):
        return False
    c = course.strip()
    return any(bool(re.match(p, c, re.IGNORECASE)) for p in RANGE_CODE_PATTERNS)

catchall_mask = (
    df["sending_course_title"].apply(is_catchall)
    | df["receiving_course_title"].apply(is_catchall)
)
range_mask = (
    df["sending_course"].apply(is_range_code)
    | df["receiving_course"].apply(is_range_code)
)
print(f"Catch-all title rows dropped  : {catchall_mask.sum()}")
print(f"Range-code course rows dropped: {range_mask.sum()}")
df = df[~catchall_mask & ~range_mask].reset_index(drop=True)

# ── 6. Drop rows still missing IPEDS IDs or subject prefixes ─────────────────

missing_mask = (
    df["receiving_institution_ipeds_id"].isna()
    | df["sending_institution_ipeds_id"].isna()
    | df["receiving_sub_pref"].isna()
    | df["sending_sub_pref"].isna()
)
print(f"Still-missing ID/sub_pref rows: {missing_mask.sum()}")
df = df[~missing_mask].reset_index(drop=True)
print(f"Clean panel: {len(df)} rows")

# ── 7. Build course graph ─────────────────────────────────────────────────────

def make_node_id(ipeds_id, sub_pref, course):
    return f"{ipeds_id.strip()}_{sub_pref.strip().upper()}_{course.strip()}"

df["sending_node"]   = df.apply(lambda r: make_node_id(r["sending_institution_ipeds_id"],   r["sending_sub_pref"],   r["sending_course"]),   axis=1)
df["receiving_node"] = df.apply(lambda r: make_node_id(r["receiving_institution_ipeds_id"], r["receiving_sub_pref"], r["receiving_course"]), axis=1)

G = nx.Graph()
for _, row in df.iterrows():
    G.add_node(row["sending_node"],
               institution=row["sending_institution"],
               ipeds_id=row["sending_institution_ipeds_id"],
               sub_pref=row["sending_sub_pref"],
               course=row["sending_course"],
               title=row["sending_course_title"],
               side="sending")
    G.add_node(row["receiving_node"],
               institution=row["receiving_institution"],
               ipeds_id=row["receiving_institution_ipeds_id"],
               sub_pref=row["receiving_sub_pref"],
               course=row["receiving_course"],
               title=row["receiving_course_title"],
               side="receiving")
    G.add_edge(row["sending_node"], row["receiving_node"])

print(f"\nGraph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# ── 8. Extract connected components ──────────────────────────────────────────

components = list(nx.connected_components(G))
print(f"Connected components: {len(components)}")

node_to_component = {}
for comp_id, comp_nodes in enumerate(components):
    for node in comp_nodes:
        node_to_component[node] = comp_id

df["component_id"] = df["sending_node"].map(node_to_component)

# ── 9. Score each component — embedding similarity ───────────────────────────

all_titles = list({
    t for t in
    pd.concat([df["sending_course_title"], df["receiving_course_title"]])
    if isinstance(t, str) and t.strip()
})

if USE_SENTENCE_TRANSFORMER:
    print("Encoding course titles with sentence-transformer (all-MiniLM-L6-v2)...")
    model      = SentenceTransformer("all-MiniLM-L6-v2")
    raw_embeds = model.encode(all_titles, batch_size=64, show_progress_bar=True,
                              convert_to_numpy=True)
    norms      = np.linalg.norm(raw_embeds, axis=1, keepdims=True)
    embeddings = raw_embeds / np.where(norms == 0, 1, norms)
    title_to_idx = {t: i for i, t in enumerate(all_titles)}

    def embedding_similarity(titles):
        valid = [t for t in titles if isinstance(t, str) and t.strip() and t in title_to_idx]
        if len(valid) < 2:
            return 1.0
        vecs = embeddings[[title_to_idx[t] for t in valid]]
        sims = vecs @ vecs.T
        n    = len(valid)
        return round(float((sims.sum() - n) / (n * (n - 1))), 3)

else:
    print("Encoding course titles with TF-IDF (fallback)...")
    word_vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)
    char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                               sublinear_tf=True, min_df=1)
    X_word = word_vec.fit_transform(all_titles)
    X_char = char_vec.fit_transform(all_titles)
    X_full = sp.hstack([X_word * 0.6, X_char * 0.4])
    title_to_idx = {t: i for i, t in enumerate(all_titles)}

    def embedding_similarity(titles):
        valid = [t for t in titles if isinstance(t, str) and t.strip() and t in title_to_idx]
        if len(valid) < 2:
            return 1.0
        vecs = X_full[[title_to_idx[t] for t in valid]]
        sims = cosine_similarity(vecs)
        n    = len(valid)
        return round(float((sims.sum() - n) / (n * (n - 1))), 3)

def subject_homogeneity(sub_prefs):
    sub_prefs = [s for s in sub_prefs if isinstance(s, str)]
    if not sub_prefs:
        return 0.0
    modal = max(set(sub_prefs), key=sub_prefs.count)
    return round(sub_prefs.count(modal) / len(sub_prefs), 3)

comp_stats = []
for comp_id, comp_nodes in enumerate(components):
    subgraph   = G.subgraph(comp_nodes)
    node_attrs = [G.nodes[n] for n in comp_nodes]
    all_t      = [a.get("title", "") for a in node_attrs]
    all_s      = [a.get("sub_pref", "") for a in node_attrs]
    all_insts  = {a.get("institution", "") for a in node_attrs}

    comp_stats.append({
        "component_id":          comp_id,
        "size":                  len(comp_nodes),
        "n_institutions":        len(all_insts),
        "n_subject_prefixes":    len(set(s for s in all_s if s)),
        "modal_subject":         max(set(all_s), key=all_s.count) if all_s else None,
        "subject_homogeneity":   subject_homogeneity(all_s),
        "mean_embed_similarity": embedding_similarity(all_t),
        "diameter":              nx.diameter(subgraph) if nx.is_connected(subgraph) else None,
    })

comp_df = pd.DataFrame(comp_stats)

# ── 10. Flag suspect components ───────────────────────────────────────────────

comp_df["flag_large"]          = comp_df["size"] > SIZE_THRESHOLD
comp_df["flag_mixed_subject"]  = comp_df["subject_homogeneity"] < HOMOGENEITY_THRESHOLD
comp_df["flag_low_similarity"] = comp_df["mean_embed_similarity"] < SIMILARITY_THRESHOLD
comp_df["flagged"] = (
    comp_df["flag_large"]
    | comp_df["flag_mixed_subject"]
    | comp_df["flag_low_similarity"]
)

n_flagged = comp_df["flagged"].sum()
n_clean   = (~comp_df["flagged"]).sum()
clean_df  = comp_df[~comp_df["flagged"]]
print(f"\nComponent quality:")
print(f"  Clean (usable for training) : {n_clean}")
print(f"  Flagged for review          : {n_flagged}")

flagged_comp_ids = set(comp_df[comp_df["flagged"]]["component_id"])
flagged_nodes    = {n for n, cid in node_to_component.items() if cid in flagged_comp_ids}
flagged_subgraph = G.subgraph(flagged_nodes)
betweenness      = nx.betweenness_centrality(flagged_subgraph, normalized=True) if flagged_nodes else {}

betweenness_df = (
    pd.DataFrame.from_dict(betweenness, orient="index", columns=["betweenness"])
    .reset_index().rename(columns={"index": "node_id"})
    .sort_values("betweenness", ascending=False)
)
betweenness_df["title"]        = betweenness_df["node_id"].map(lambda n: G.nodes[n].get("title", "") if n in G.nodes else "")
betweenness_df["institution"]  = betweenness_df["node_id"].map(lambda n: G.nodes[n].get("institution", "") if n in G.nodes else "")
betweenness_df["component_id"] = betweenness_df["node_id"].map(node_to_component)

print(f"\nTop 10 high-betweenness nodes in flagged components:")
print(
    betweenness_df[betweenness_df["betweenness"] > 0]
    .head(10)[["node_id", "title", "institution", "betweenness"]]
    .to_string(index=False)
)

# ── 11. Save outputs ──────────────────────────────────────────────────────────

panel_out = df.merge(
    comp_df[["component_id", "flagged", "subject_homogeneity", "mean_embed_similarity"]],
    on="component_id", how="left",
)
panel_out["component_id"] = panel_out["component_id"].astype(str)
panel_out.to_csv(OUT_PANEL, index=False)
print(f"\nSaved: {OUT_PANEL}  ({len(panel_out)} rows)")

comp_df.to_csv(OUT_COMPONENTS, index=False)
print(f"Saved: {OUT_COMPONENTS}  ({len(comp_df)} components)")

review_out = panel_out[panel_out["flagged"]].sort_values(["component_id", "mean_embed_similarity"])
review_out.to_csv(OUT_REVIEW, index=False)
print(f"Saved: {OUT_REVIEW}  ({len(review_out)} rows across {n_flagged} flagged components)")

print("\nDone.")

# ── 12. Visualizations ────────────────────────────────────────────────────────

OUT_VIZ_OVERVIEW  = "data/viz_overview.png"
OUT_VIZ_SUBJECTS  = "data/viz_subjects.png"
OUT_VIZ_COMP_GRID = "data/viz_component_grid.png"
OUT_VIZ_COMPS_DIR = "data/component_maps"
os.makedirs(OUT_VIZ_COMPS_DIR, exist_ok=True)

CLEAN_COL = "#4ade80"
FLAG_COL  = "#f87171"
NEUT_COL  = "#94a3b8"
BG_COL    = "#1e2330"
GRID_COL  = "#2d3447"

def style_ax(ax, title):
    ax.set_facecolor(BG_COL)
    ax.set_title(title, color="white", fontsize=11, pad=8)
    ax.tick_params(colors=NEUT_COL, labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COL)
    ax.yaxis.label.set_color(NEUT_COL)
    ax.xaxis.label.set_color(NEUT_COL)
    ax.grid(color=GRID_COL, linewidth=0.5, alpha=0.7)

# ── 12a. Overview ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 12), facecolor="#0f1117")
fig.suptitle("NJ Transfer Network — Component Quality Overview",
             fontsize=16, fontweight="bold", color="white", y=0.98)

flag_df   = comp_df[comp_df["flagged"]]
sim_label = f"{'0.45' if USE_SENTENCE_TRANSFORMER else '0.3'}"

ax1 = fig.add_subplot(2, 2, 1)
style_ax(ax1, "Embed Similarity vs Subject Homogeneity")
ax1.scatter(flag_df["subject_homogeneity"],  flag_df["mean_embed_similarity"],
            c=FLAG_COL, alpha=0.35, s=18, label=f"Flagged ({len(flag_df)})", linewidths=0)
ax1.scatter(clean_df["subject_homogeneity"], clean_df["mean_embed_similarity"],
            c=CLEAN_COL, alpha=0.8, s=28, label=f"Clean ({len(clean_df)})", linewidths=0)
ax1.axhline(SIMILARITY_THRESHOLD,  color=FLAG_COL, lw=1, ls="--", alpha=0.6)
ax1.axvline(HOMOGENEITY_THRESHOLD, color=FLAG_COL, lw=1, ls="--", alpha=0.6)
ax1.set_xlabel("Subject Homogeneity")
ax1.set_ylabel("Mean Embedding Similarity")
ax1.legend(fontsize=8, facecolor=BG_COL, labelcolor="white", framealpha=0.8)

ax2 = fig.add_subplot(2, 2, 2)
style_ax(ax2, "Component Size Distribution")
bins = range(2, 22)
ax2.hist(flag_df["size"].clip(upper=20),  bins=bins, color=FLAG_COL,  alpha=0.6, label="Flagged", edgecolor="none")
ax2.hist(clean_df["size"].clip(upper=20), bins=bins, color=CLEAN_COL, alpha=0.8, label="Clean",   edgecolor="none")
ax2.set_xlabel("Component Size (nodes, capped at 20)")
ax2.set_ylabel("Count")
ax2.legend(fontsize=8, facecolor=BG_COL, labelcolor="white", framealpha=0.8)

ax3 = fig.add_subplot(2, 2, 3)
style_ax(ax3, "Flag Reason Breakdown")
reasons = {
    "Large\n(size > 10)":                       int(comp_df["flag_large"].sum()),
    "Mixed subject\n(hom < 0.8)":               int(comp_df["flag_mixed_subject"].sum()),
    f"Low similarity\n(sim < {sim_label})":     int(comp_df["flag_low_similarity"].sum()),
    "Clean":                                     int((~comp_df["flagged"]).sum()),
}
colors = [FLAG_COL, FLAG_COL, FLAG_COL, CLEAN_COL]
bars = ax3.bar(reasons.keys(), reasons.values(), color=colors, alpha=0.85, width=0.5)
for bar, val in zip(bars, reasons.values()):
    ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
             str(val), ha="center", va="bottom", color="white", fontsize=9)
ax3.set_ylabel("Number of Components")
ax3.set_ylim(0, max(reasons.values()) * 1.15)

ax4 = fig.add_subplot(2, 2, 4)
style_ax(ax4, "Largest Clean Component")
ax4.grid(False)
largest_clean_id = str(int(clean_df.sort_values("size", ascending=False).iloc[0]["component_id"]))
comp_rows = panel_out[panel_out["component_id"] == largest_clean_id]
H = nx.Graph()
for _, row in comp_rows.iterrows():
    for side in ("sending", "receiving"):
        nid = row[f"{side}_node"]
        if nid not in H:
            H.add_node(nid, label=f"{row[f'{side}_sub_pref']} {row[f'{side}_course']}",
                       title=row[f"{side}_course_title"], side=side)
    H.add_edge(row["sending_node"], row["receiving_node"])
pos = nx.spring_layout(H, seed=42, k=1.5)
node_colors = [CLEAN_COL if H.nodes[n]["side"] == "sending" else "#60a5fa" for n in H.nodes]
nx.draw_networkx_edges(H, pos, ax=ax4, edge_color=NEUT_COL, alpha=0.5, width=1.5)
nx.draw_networkx_nodes(H, pos, ax=ax4, node_color=node_colors, node_size=320, alpha=0.95)
nx.draw_networkx_labels(H, pos, ax=ax4, labels={n: H.nodes[n]["label"] for n in H.nodes},
                        font_size=6, font_color="white")
ax4.legend(handles=[
    mpatches.Patch(color=CLEAN_COL, label="Sending"),
    mpatches.Patch(color="#60a5fa",  label="Receiving"),
], fontsize=7, facecolor=BG_COL, labelcolor="white", framealpha=0.8)

plt.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(OUT_VIZ_OVERVIEW, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close(fig)
print(f"Saved: {OUT_VIZ_OVERVIEW}")

# ── 12b. Subject clean-rate bar chart ─────────────────────────────────────────
subj_stats = (
    comp_df.groupby("modal_subject")
    .agg(total=("component_id", "count"), clean=("flagged", lambda x: (~x).sum()),
         mean_sim=("mean_embed_similarity", "mean"))
    .assign(clean_rate=lambda d: d["clean"] / d["total"])
    .query("total >= 3")
    .sort_values("clean_rate", ascending=False)
    .head(30).reset_index()
)
fig2, ax = plt.subplots(figsize=(12, 8), facecolor="#0f1117")
ax.set_facecolor(BG_COL)
fig2.suptitle("Clean Component Rate by Subject Prefix (subjects with ≥ 3 components)",
              fontsize=13, fontweight="bold", color="white", y=1.01)
bar_colors = [CLEAN_COL if r >= 0.5 else FLAG_COL for r in subj_stats["clean_rate"]]
bars = ax.barh(subj_stats["modal_subject"], subj_stats["clean_rate"],
               color=bar_colors, alpha=0.85, height=0.65)
for bar, row in zip(bars, subj_stats.itertuples()):
    ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
            f"{int(row.clean)}/{int(row.total)}", va="center", ha="left",
            fontsize=7, color=NEUT_COL)
ax.set_xlabel("Fraction of components passing quality filters", color=NEUT_COL)
ax.set_xlim(0, 1.18)
ax.axvline(0.5, color=NEUT_COL, lw=1, ls="--", alpha=0.5)
ax.tick_params(colors=NEUT_COL, labelsize=8)
for spine in ax.spines.values():
    spine.set_edgecolor(GRID_COL)
ax.grid(axis="x", color=GRID_COL, linewidth=0.5, alpha=0.7)
ax.invert_yaxis()
plt.tight_layout()
fig2.savefig(OUT_VIZ_SUBJECTS, dpi=150, bbox_inches="tight", facecolor=fig2.get_facecolor())
plt.close(fig2)
print(f"Saved: {OUT_VIZ_SUBJECTS}")

# ── 12c. Component network maps ───────────────────────────────────────────────

COMMUNITY_PALETTE = [
    "#4a90d9", "#e07b39", "#5bb56e", "#c75bbd",
    "#c4a230", "#5bbdbd", "#d95f6e", "#8c7adb",
]

def build_component_graph(component_id, panel_df):
    rows = panel_df[panel_df["component_id"] == str(component_id)]
    H = nx.Graph()
    for _, r in rows.iterrows():
        for side in ("sending", "receiving"):
            nid = r[f"{side}_node"]
            if nid not in H:
                subj   = r[f"{side}_sub_pref"] or ""
                course = r[f"{side}_course"] or ""
                title  = r[f"{side}_course_title"] or ""
                H.add_node(nid, label=f"{subj} {course} | {title[:20]}",
                           sub_pref=subj, title=title,
                           institution=r[f"{side}_institution"], side=side)
        H.add_edge(r["sending_node"], r["receiving_node"])
    return H

def plot_component(component_id, panel_df, ax=None, title_prefix=""):
    H = build_component_graph(component_id, panel_df)
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(12, 8), facecolor="white")
        ax.set_facecolor("white")

    communities    = list(greedy_modularity_communities(H))
    node_community = {node: i for i, comm in enumerate(communities) for node in comm}
    node_colors    = [COMMUNITY_PALETTE[node_community.get(n, 0) % len(COMMUNITY_PALETTE)]
                      for n in H.nodes]

    pos = nx.spring_layout(H, seed=42, k=2.5 / max(len(H.nodes) ** 0.5, 1))
    nx.draw_networkx_edges(H, pos, ax=ax, edge_color="#cccccc", width=1.2, alpha=0.9)
    nx.draw_networkx_nodes(H, pos, ax=ax, node_color=node_colors, node_size=120, alpha=1.0)
    nx.draw_networkx_labels(H, pos, ax=ax,
                            labels={n: H.nodes[n]["label"] for n in H.nodes},
                            font_size=7, font_color="#222222", font_family="sans-serif")
    ax.set_title(
        f"{title_prefix}Disciplinary Component Map  (Color = Infomap Flow Module)\n"
        f"Nodes: {H.number_of_nodes()} | Edges: {H.number_of_edges()}",
        fontsize=10, color="#222222", pad=10,
    )
    ax.axis("off")
    ax.set_facecolor("white")

    if standalone:
        fig.patch.set_facecolor("white")
        return fig

COMPONENT_IDS_TO_PLOT = (
    clean_df.sort_values("size", ascending=False)
    .head(6)["component_id"].astype(int).tolist()
)

for cid in COMPONENT_IDS_TO_PLOT:
    fig = plot_component(cid, panel_out, title_prefix=f"Component {cid} — ")
    out_path = f"{OUT_VIZ_COMPS_DIR}/component_{cid}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path}")

fig_grid, axes = plt.subplots(2, 3, figsize=(20, 12), facecolor="white")
fig_grid.suptitle("NJ Transfer Network — Top 6 Clean Components",
                  fontsize=14, fontweight="bold", color="#222222", y=1.01)
for ax, cid in zip(axes.flat, COMPONENT_IDS_TO_PLOT):
    plot_component(cid, panel_out, ax=ax, title_prefix=f"ID {cid} — ")
plt.tight_layout()
fig_grid.savefig(OUT_VIZ_COMP_GRID, dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig_grid)
print(f"Saved: {OUT_VIZ_COMP_GRID}")