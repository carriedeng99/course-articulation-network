"""
nj_baseline_eval.py
────────────────────────────────────────────────────────────────────────────────
Evaluate the base Qwen3-Embedding-8B model (no fine-tuning) on the NJ test split.
Uses the same Recall@k metric as nj_finetune_train.py so results are comparable.

For each test pair (sending_text, receiving_text):
  - Embed all unique receiving-side texts
  - Rank by cosine similarity to the sending embedding
  - Report Recall@1, Recall@5, Recall@10 and mean cosine similarity of true pairs

Usage
-----
    python nj_baseline_eval.py \
        --test  nj_ft_data/nj_test.jsonl \
        --model Qwen/Qwen3-Embedding-8B
"""

import argparse
import json

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--test",  required=True, help="Path to nj_test.jsonl")
parser.add_argument("--model", default="Qwen/Qwen3-Embedding-8B")
parser.add_argument("--batch_size", type=int, default=128)
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# ── Load test pairs ───────────────────────────────────────────────────────────
with open(args.test) as f:
    test_pairs = [json.loads(line) for line in f]
print(f"Test pairs: {len(test_pairs)}")

# ── Load model ────────────────────────────────────────────────────────────────
print(f"\nLoading model: {args.model}")
model = SentenceTransformer(args.model, device=device)
model.eval()

# ── Embed ─────────────────────────────────────────────────────────────────────
print("Embedding receiving-side courses...")
recv_texts = list({p["receiving_text"] for p in test_pairs})
recv_index = {t: i for i, t in enumerate(recv_texts)}

recv_embeddings = model.encode(
    recv_texts, batch_size=args.batch_size,
    convert_to_tensor=True, normalize_embeddings=True,
    show_progress_bar=True,
)

print("Embedding sending-side courses...")
send_texts = [p["sending_text"] for p in test_pairs]
send_embeddings = model.encode(
    send_texts, batch_size=args.batch_size,
    convert_to_tensor=True, normalize_embeddings=True,
    show_progress_bar=True,
)

# ── Score ─────────────────────────────────────────────────────────────────────
print("Computing similarity matrix...")
sim_matrix = torch.mm(
    send_embeddings.float(), recv_embeddings.float().T
).cpu().numpy()   # [N_pairs, N_recv]

hits = {1: 0, 5: 0, 10: 0}
true_sims = []

for i, pair in enumerate(test_pairs):
    true_idx = recv_index[pair["receiving_text"]]
    true_sim = sim_matrix[i, true_idx]
    true_sims.append(float(true_sim))

    rank = int((sim_matrix[i] >= true_sim).sum())   # 1-indexed rank
    for k in [1, 5, 10]:
        if rank <= k:
            hits[k] += 1

n = len(test_pairs)

# ── Report ────────────────────────────────────────────────────────────────────
print("\n── Baseline results (no fine-tuning) ───────────────────────────────────")
print(f"  Model:          {args.model}")
print(f"  Test pairs:     {n}")
print(f"  Recall@1:       {hits[1]/n:.4f}  ({hits[1]}/{n})")
print(f"  Recall@5:       {hits[5]/n:.4f}  ({hits[5]}/{n})")
print(f"  Recall@10:      {hits[10]/n:.4f}  ({hits[10]}/{n})")
print(f"  Mean true sim:  {np.mean(true_sims):.4f}")
print(f"  Min true sim:   {np.min(true_sims):.4f}")
print(f"  Max true sim:   {np.max(true_sims):.4f}")
print("────────────────────────────────────────────────────────────────────────")
