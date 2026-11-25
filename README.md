# Course Articulation Network

A graph-based pipeline for collecting, cleaning, and embedding course transfer equivalency data across U.S. state higher education systems. Built to support research on **higher education equity and transfer credit policy**, with applications to understanding how institutional articulation agreements shape student mobility across colleges and universities.

## Overview

When a student transfers between colleges, their credits are evaluated against a receiving institution's curriculum through **articulation agreements** — formal declarations that course A at institution X is equivalent to course B at institution Y. Access to clear, consistent transfer pathways has well-documented effects on degree completion, time-to-graduation, and equity for community college students — yet the underlying data are scattered across dozens of state portals in inconsistent formats.

This project structures those agreements into a bipartite course graph, uses graph-connected components as supervision signal for fine-tuning sentence embedding models, and flags semantically inconsistent equivalencies for expert review. The result is a reproducible, state-scalable pipeline that transforms raw articulation tables into analysis-ready data for quantitative social science research.

The pipeline has been applied to six states: **New Jersey, Tennessee, Wyoming, Minnesota, Idaho, and Arkansas**, covering hundreds of institutions and tens of thousands of articulation pairs.

---

## Pipeline

```
State transfer portal (e.g. njtransfer.org)
     │
     ▼
0. Web scraping (scripts/scrape/nj_scrape.py)
   ├── Enumerate all sending × receiving institution pairs
   ├── Loop keyword search (a–z, 0–9) to collect course-pair URLs
   ├── Parse each course-pair page: sending/receiving course codes,
   │   titles, GenEd areas, notes, syllabus links
   ├── Deduplicate by course_id (keep most recent start date)
   └── Outputs: data/urls/*.csv, data/jsons/*.json, data/panel.csv
     │
     ▼
0b. IPEDS institution matching (scripts/scrape/match_ipeds.py)
   ├── Fuzzy-match scraped institution names → IPEDS unit IDs
   │   using the IPEDS Header Data file (hd2024.csv)
   ├── Matching cascade: exact → fuzzy (RapidFuzz) → geo tiebreak
   │   → active-status tiebreak
   ├── Sub-school groupings (e.g. Rutgers campuses) manually confirmed
   │   and recorded in manual_review_resolved.csv
   └── Outputs: matches.csv, unmatched.csv (for manual review)
     │
     ▼
1. Network construction (scripts/network/<state>_network.py)
   ├── Merge resolved IPEDS IDs (manual_review_resolved.csv)
   ├── Deduplication
   ├── Subject prefix normalization (SUBJ_NORM)
   ├── Drop catch-all equivalencies (range codes, internship buckets)
   ├── Bipartite graph construction
   ├── Connected component extraction → contrastive learning groups
   ├── Embedding similarity scoring (Qwen/Qwen3-Embedding-8B)
   ├── Flagging suspect components (low similarity, high betweenness)
   └── Outputs: panel_clean.csv, review_flagged.csv, visualizations
     │
     ▼
2. Fine-tuning data prep (scripts/finetune/nj_finetune_prep.py)
   ├── 90/10 train/test split at the pair level
   ├── Strict node exclusion: test nodes removed from train entirely
   └── Outputs: nj_train.jsonl, nj_test.jsonl
     │
     ▼
3. Baseline evaluation (scripts/finetune/nj_baseline_eval.py)
   └── Recall@1/5/10 on test set using the base (unfinetuned) model
     │
     ▼
4. Supervised contrastive fine-tuning (scripts/finetune/nj_finetune_train.py)
   ├── Loss: SupConLoss — connected components as positive sets,
   │         other components in batch as negatives
   ├── Component-balanced batch sampler
   ├── 8-bit AdamW + gradient checkpointing for memory efficiency
   └── Recall@1/5/10 evaluated after each epoch; best model saved
```

---

## NJ Example

New Jersey is used as the worked example throughout this repo. The NJ transfer portal ([njtransfer.org](https://njtransfer.org)) covers articulation agreements across ~40 institutions.

**After cleaning:**

| Metric | Value |
|--------|-------|
| Pairs (edges) | 3,784 |
| Unique course nodes | 6,626 |
| Connected components | 2,842 |
| Flagged components | 515 (13.6%) |
| Mean embedding similarity | 0.816 |

**Example visualizations** (see `outputs/nj/`):

- `viz_overview.png` — graph-level statistics and similarity distribution
- `viz_subjects.png` — subject-area breakdown of articulation pairs
- `viz_component_grid.png` — top clean components by size

**Flagging logic:** Components are flagged when their mean pairwise embedding similarity falls below the 10th percentile, indicating potentially erroneous or semantically inconsistent equivalencies. High-betweenness nodes (top 10%) are flagged separately as structural bridge nodes that may indicate data quality issues.

---

## Embedding Model

All courses are embedded using **[Qwen/Qwen3-Embedding-8B](https://huggingface.co/Qwen/Qwen3-Embedding-8B)** via `sentence-transformers`.

Embed text format:
```
<subject_prefix> <course_number> <course_title>
```
Example: `MAT 151 CALCULUS I`

---

## Fine-Tuning

Fine-tuning uses **Supervised Contrastive Loss** (Khosla et al. 2020):
- **Positives**: all course pairs sharing a connected component (confirmed transfer equivalencies)
- **Negatives**: all other courses in the same batch (implicit, in-batch)

The train/test split enforces strict node exclusion — any node appearing in a test pair is removed from training entirely, preventing leakage.

Evaluation metric: **Recall@k** — for each test pair, rank all receiving-side courses by cosine similarity to the sending course and check whether the true equivalency appears in the top k.

---

## Repo Structure

```
course-articulation-network/
├── README.md
├── environment.yml               # Conda environment
├── requirements.txt              # pip requirements
├── data/
│   └── sample/
│       └── nj_panel_sample.csv  # 50-row NJ sample (raw/url columns removed)
├── scripts/
│   ├── scrape/                   # Step 0: web scraping + IPEDS matching
│   │   ├── nj_scrape.py          # Scrape NJTransfer portal → panel.csv
│   │   └── match_ipeds.py        # Fuzzy-match institution names → IPEDS IDs
│   ├── network/                  # Step 1: state network construction
│   │   ├── nj_network.py
│   │   ├── tn_network.py
│   │   ├── mn_network.py
│   │   ├── wy_network.py
│   │   ├── id_network.py
│   │   └── ar_network.py
│   ├── finetune/                 # Steps 2–4: fine-tuning pipeline (NJ example)
│   │   ├── nj_finetune_prep.py
│   │   ├── nj_finetune_train.py
│   │   └── nj_baseline_eval.py
│   └── slurm/                    # SLURM batch job scripts (Oscar HPC, Brown University)
│       ├── run_tn.sh
│       ├── run_wy.sh
│       ├── nj_finetune_train.sh
│       └── nj_baseline_eval.sh
└── outputs/
    └── nj/                       # Example outputs from the NJ run
        ├── viz_overview.png
        ├── viz_subjects.png
        └── viz_component_grid.png
```

---

## Setup

```bash
# Clone
git clone https://github.com/carriedeng99/course-articulation-network.git
cd course-articulation-network

# Create environment (conda)
conda env create -f environment.yml
conda activate transfer_net

# Or pip
pip install -r requirements.txt
```

The embedding model (`Qwen/Qwen3-Embedding-8B`) downloads automatically from Hugging Face on first run. A GPU is strongly recommended — the network scripts were run on an NVIDIA L40S (44GB VRAM) via Brown University's Oscar HPC cluster.

---

## Running the NJ Pipeline

```bash
# 0. Scrape NJTransfer portal → panel.csv
#    (requires requests, beautifulsoup4, tqdm; ~hours to run, respects rate limits)
python scripts/scrape/nj_scrape.py

# 0b. Match institution names to IPEDS unit IDs
#     (requires rapidfuzz; download hd2024.csv from IPEDS Data Center)
python scripts/scrape/match_ipeds.py \
    --panel data/panel.csv \
    --hd    hd2024.csv \
    --state NJ \
    --out   ./
#    Review unmatched.csv; add confirmed sub-schools to manual_review_resolved.csv

# 1. Build network, embed courses, flag components
python scripts/network/nj_network.py

# 2. Prep train/test split
python scripts/finetune/nj_finetune_prep.py \
    --panel data/panel_clean.csv \
    --out_dir ./nj_ft_data

# 3. Baseline evaluation (no fine-tuning)
python scripts/finetune/nj_baseline_eval.py \
    --test nj_ft_data/nj_test.jsonl

# 4. Fine-tune
python scripts/finetune/nj_finetune_train.py \
    --train nj_ft_data/nj_train.jsonl \
    --test  nj_ft_data/nj_test.jsonl \
    --out_dir ./nj_finetuned \
    --epochs 5
```

---

## Data

Raw panel CSVs are not included in this repo (data use agreement with state transfer portals). The `data/sample/nj_panel_sample.csv` file contains 50 anonymized rows illustrating the schema.

**Panel schema:**

| Column | Description |
|--------|-------------|
| `sending_institution` | Name of the sending college |
| `sending_institution_ipeds_id` | IPEDS unit ID |
| `sending_sub_pref` | Subject prefix (raw) |
| `sending_course` | Course number |
| `sending_course_title` | Course title |
| `receiving_*` | Same fields for the receiving institution |
| `sending_embed_text` | Text used for embedding: `<prefix> <number> <title>` |
| `component_id` | Connected component ID (contrastive learning label) |
| `mean_embed_similarity` | Mean cosine similarity within component |
| `flagged` | Whether component is flagged for review |

---

## Citation / Acknowledgements

Built as part of a research project on higher education transfer credit policy at Brown University Annenberg Institute, project led by Dr. Anna Paulson. Embedding model: [Qwen/Qwen3-Embedding-8B](https://huggingface.co/Qwen/Qwen3-Embedding-8B). Compute provided by Brown University's Oscar HPC cluster.