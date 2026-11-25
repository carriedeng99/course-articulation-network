"""
nj_finetune_prep.py
────────────────────────────────────────────────────────────────────────────────
Prepare NJ transfer-network data for supervised contrastive fine-tuning.

Pipeline
--------
1. Load panel_clean.csv
2. Build a flat node table: one row per unique course node, with its
   embed_text and component_id label.
3. 90 / 10 train / test split at the *pair* level, with strict node
   exclusion: any node that appears in a test pair is removed from train.
4. Write:
     nj_train.jsonl  — {"text": <embed_text>, "label": <component_id>}
     nj_test.jsonl   — {"text": <embed_text>, "label": <component_id>,
                         "node_id": <node_id>,
                         "sending_node": ..., "receiving_node": ...}
                       (test keeps pair structure for Recall@k eval)

Usage
-----
    python nj_finetune_prep.py \
        --panel panel_clean.csv \
        --out_dir ./ \
        --seed 42

Output stats are printed so you can sanity-check before training.
"""

import argparse
import json
import random
from pathlib import Path

import pandas as pd

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--panel",   required=True,  help="Path to panel_clean.csv")
parser.add_argument("--out_dir", default="./",   help="Directory for output JSONL files")
parser.add_argument("--seed",    type=int, default=42)
args = parser.parse_args()

random.seed(args.seed)
out_dir = Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)

# ── 1. Load ───────────────────────────────────────────────────────────────────
df = pd.read_csv(args.panel)
print(f"Loaded panel: {len(df)} rows, {df['component_id'].nunique()} components")

# ── 2. Build flat node table ──────────────────────────────────────────────────
# Each edge (row) contributes two nodes.  We keep the richer embed_text and
# component_id for each; duplicates are collapsed.

send_nodes = df[["sending_node", "sending_embed_text", "component_id"]].rename(
    columns={"sending_node": "node_id", "sending_embed_text": "text"}
)
recv_nodes = df[["receiving_node", "receiving_embed_text", "component_id"]].rename(
    columns={"receiving_node": "node_id", "receiving_embed_text": "text"}
)
nodes = (
    pd.concat([send_nodes, recv_nodes], ignore_index=True)
    .drop_duplicates(subset="node_id")
    .reset_index(drop=True)
)
print(f"Unique nodes: {len(nodes)}")

# ── 3. 90 / 10 split at pair level, with node exclusion ──────────────────────
# Shuffle pairs, take 10 % as test.  Remove any training node that appears
# in a test pair — this prevents the model from seeing test-set courses
# during training even as the "other side" of a different pair.

pairs = df[["sending_node", "receiving_node", "component_id"]].copy()
pairs = pairs.sample(frac=1, random_state=args.seed).reset_index(drop=True)

n_test = max(1, int(round(len(pairs) * 0.10)))
test_pairs  = pairs.iloc[:n_test].copy()
train_pairs = pairs.iloc[n_test:].copy()

# Collect all nodes that appear in test pairs
test_nodes = set(test_pairs["sending_node"]) | set(test_pairs["receiving_node"])

print(f"\nPair split: {len(train_pairs)} train / {len(test_pairs)} test")
print(f"Test nodes (excluded from train): {len(test_nodes)}")

# ── 4. Build train node set (exclude test nodes) ──────────────────────────────
train_node_ids = (
    set(train_pairs["sending_node"]) | set(train_pairs["receiving_node"])
) - test_nodes

train_nodes = nodes[nodes["node_id"].isin(train_node_ids)].copy()
test_nodes_df = nodes[nodes["node_id"].isin(test_nodes)].copy()

# How many components survive in train vs test?
train_comps = train_nodes["component_id"].nunique()
test_comps  = test_nodes_df["component_id"].nunique()
print(f"\nTrain: {len(train_nodes)} nodes across {train_comps} components")
print(f"Test:  {len(test_nodes_df)} nodes across {test_comps} components")

# Component size distribution in train
train_comp_sizes = train_nodes.groupby("component_id").size()
print(f"\nTrain component sizes:")
print(f"  size 1 (singletons): {(train_comp_sizes == 1).sum()}")
print(f"  size 2:              {(train_comp_sizes == 2).sum()}")
print(f"  size 3-5:            {((train_comp_sizes >= 3) & (train_comp_sizes <= 5)).sum()}")
print(f"  size 6+:             {(train_comp_sizes >= 6).sum()}")
print(f"  mean size:           {train_comp_sizes.mean():.2f}")

# Singletons won't contribute positives to SupConLoss but we keep them
# in the file; the batch sampler in training will deprioritise them.
n_singleton_train = (train_comp_sizes == 1).sum()
n_multi_train     = (train_comp_sizes >= 2).sum()
print(f"\n  Multi-member components (usable for SupCon): {n_multi_train}")
print(f"  Singleton components (no within-batch positives): {n_singleton_train}")

# ── 5. Write JSONL ────────────────────────────────────────────────────────────

# Train: one record per node
train_path = out_dir / "nj_train.jsonl"
with open(train_path, "w") as f:
    for _, row in train_nodes.iterrows():
        record = {
            "text":         row["text"],
            "label":        int(row["component_id"]),
            "node_id":      row["node_id"],
        }
        f.write(json.dumps(record) + "\n")
print(f"\nWrote {len(train_nodes)} records → {train_path}")

# Test: keep pair structure for Recall@k evaluation
# Each test pair becomes two records linked by pair_id so eval can rank
test_path = out_dir / "nj_test.jsonl"
test_pair_records = []
for pair_idx, row in test_pairs.iterrows():
    send_text = nodes.loc[nodes["node_id"] == row["sending_node"], "text"].values
    recv_text  = nodes.loc[nodes["node_id"] == row["receiving_node"], "text"].values
    if len(send_text) == 0 or len(recv_text) == 0:
        continue
    test_pair_records.append({
        "pair_id":        int(pair_idx),
        "component_id":   int(row["component_id"]),
        "sending_node":   row["sending_node"],
        "receiving_node": row["receiving_node"],
        "sending_text":   send_text[0],
        "receiving_text": recv_text[0],
    })

with open(test_path, "w") as f:
    for rec in test_pair_records:
        f.write(json.dumps(rec) + "\n")
print(f"Wrote {len(test_pair_records)} pairs → {test_path}")

# ── 6. Summary ────────────────────────────────────────────────────────────────
print("\n── Summary ──────────────────────────────────────────────────────────────")
print(f"  Total pairs:              {len(df)}")
print(f"  Train pairs (edge count): {len(train_pairs)}")
print(f"  Test pairs:               {len(test_pair_records)}")
print(f"  Train nodes:              {len(train_nodes)}")
print(f"  Test nodes:               {len(test_nodes_df)}")
print(f"  Node overlap (should=0):  {len(set(train_nodes['node_id']) & test_nodes)}")
print(f"  Seed:                     {args.seed}")
print("Done.")
