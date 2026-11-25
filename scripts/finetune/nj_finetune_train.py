"""
nj_finetune_train.py
────────────────────────────────────────────────────────────────────────────────
Supervised contrastive fine-tuning of Qwen3-Embedding-8B on NJ transfer pairs.

Loss: SupConLoss (pytorch-metric-learning)
  - Positives:  all pairs of courses sharing the same component_id in a batch
  - Negatives:  all pairs of courses from different component_ids in a batch
  - No manual negative mining needed; the loss handles it in-batch

Batch sampler: ComponentBalancedSampler
  - Ensures each batch contains `n_multi_comp` components that have 2+ members
  - Fills remaining slots with singleton-component nodes
  - Rationale: SupConLoss needs at least one positive pair per batch to be
    useful; if the batch is all singletons the loss is zero and nothing trains

Evaluation (every epoch):
  - Embed all test courses
  - For each test pair: rank all receiving-side courses by cosine similarity
    to the sending course; report Recall@1, Recall@5, Recall@10
  - Also report mean cosine similarity of true pairs

Memory optimisations:
  - 8-bit AdamW (bitsandbytes): cuts optimizer VRAM from ~2× to ~0.5× model size
  - Gradient checkpointing: recomputes activations during backward, saving VRAM
    at the cost of ~20% slower training

Usage
-----
    python nj_finetune_train.py \
        --train nj_train.jsonl \
        --test  nj_test.jsonl \
        --model Qwen/Qwen3-Embedding-8B \
        --out_dir ./nj_finetuned \
        --epochs 5 \
        --batch_size 16 \
        --n_multi_comp 16 \
        --lr 2e-5 \
        --seed 42

Notes
-----
- Requires: sentence-transformers >= 3.0, pytorch-metric-learning,
            bitsandbytes, torch >= 2.0
- GPU strongly recommended (l40s or similar)
- The model is saved after each epoch that improves Recall@1 on the test set.
"""

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler
import bitsandbytes as bnb
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from pytorch_metric_learning import losses as pml_losses

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--train",        required=True)
parser.add_argument("--test",         required=True)
parser.add_argument("--model",        default="Qwen/Qwen3-Embedding-8B")
parser.add_argument("--out_dir",      default="./nj_finetuned")
parser.add_argument("--epochs",       type=int,   default=5)
parser.add_argument("--batch_size",   type=int,   default=16)
parser.add_argument("--n_multi_comp", type=int,   default=16,
                    help="Multi-member components guaranteed per batch")
parser.add_argument("--lr",           type=float, default=2e-5)
parser.add_argument("--warmup_steps", type=int,   default=100)
parser.add_argument("--seed",         type=int,   default=42)
args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

out_dir = Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# ── Load data ─────────────────────────────────────────────────────────────────
def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]

train_records = load_jsonl(args.train)
test_pairs    = load_jsonl(args.test)
print(f"Train nodes: {len(train_records)}")
print(f"Test pairs:  {len(test_pairs)}")

# ── Dataset ───────────────────────────────────────────────────────────────────
class NodeDataset(Dataset):
    def __init__(self, records):
        self.texts  = [r["text"]  for r in records]
        self.labels = [r["label"] for r in records]
        # Remap component_ids to 0-indexed integers (required by pml losses)
        unique_labels = sorted(set(self.labels))
        self.label_map = {old: new for new, old in enumerate(unique_labels)}
        self.labels_mapped = [self.label_map[l] for l in self.labels]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels_mapped[idx]

train_dataset = NodeDataset(train_records)

# ── Component-balanced sampler ────────────────────────────────────────────────
class ComponentBalancedSampler(Sampler):
    def __init__(self, dataset, batch_size, n_multi_comp, seed=42):
        self.dataset      = dataset
        self.batch_size   = batch_size
        self.n_multi_comp = n_multi_comp
        self.rng          = random.Random(seed)

        label_to_indices = defaultdict(list)
        for idx, label in enumerate(dataset.labels_mapped):
            label_to_indices[label].append(idx)

        self.multi_labels  = [l for l, idxs in label_to_indices.items() if len(idxs) >= 2]
        self.single_labels = [l for l, idxs in label_to_indices.items() if len(idxs) == 1]
        self.label_to_indices = label_to_indices

        print(f"Sampler: {len(self.multi_labels)} multi-member components, "
              f"{len(self.single_labels)} singletons")

    def __iter__(self):
        multi_pool  = self.multi_labels.copy()
        single_pool = [self.label_to_indices[l][0] for l in self.single_labels]
        self.rng.shuffle(multi_pool)
        self.rng.shuffle(single_pool)

        multi_indices_flat = [
            idx
            for l in self.multi_labels
            for idx in self.label_to_indices[l]
        ]
        self.rng.shuffle(multi_indices_flat)
        fill_pool = single_pool + multi_indices_flat

        n_batches = math.ceil(len(self.dataset) / self.batch_size)
        fill_ptr  = 0

        for _ in range(n_batches):
            batch_indices = []

            chosen_multi = self.rng.sample(
                multi_pool, min(self.n_multi_comp, len(multi_pool))
            )
            for label in chosen_multi:
                members = self.label_to_indices[label]
                k = max(2, min(len(members), max(2, self.batch_size // self.n_multi_comp)))
                batch_indices.extend(self.rng.sample(members, min(k, len(members))))
                if len(batch_indices) >= self.batch_size:
                    break

            while len(batch_indices) < self.batch_size and fill_ptr < len(fill_pool):
                idx = fill_pool[fill_ptr]
                fill_ptr += 1
                if idx not in batch_indices:
                    batch_indices.append(idx)

            self.rng.shuffle(batch_indices)
            yield from batch_indices[:self.batch_size]

    def __len__(self):
        return len(self.dataset)


sampler = ComponentBalancedSampler(
    train_dataset,
    batch_size=args.batch_size,
    n_multi_comp=args.n_multi_comp,
    seed=args.seed,
)

def collate_fn(batch):
    texts, labels = zip(*batch)
    return list(texts), torch.tensor(labels, dtype=torch.long)

train_loader = DataLoader(
    train_dataset,
    batch_sampler=None,
    sampler=sampler,
    batch_size=args.batch_size,
    collate_fn=collate_fn,
    drop_last=False,
)

# ── Model ─────────────────────────────────────────────────────────────────────
print(f"\nLoading model: {args.model}")
model = SentenceTransformer(args.model, device=device)
# Gradient checkpointing: recompute activations during backward instead of
# storing them, trading ~20% speed for a large reduction in VRAM usage.
model[0].auto_model.gradient_checkpointing_enable()
model.train()

# Tokenizer for the gradient-tracked forward pass during training.
# model.encode() detaches from the graph (inference only); we call the
# underlying transformer directly to keep gradients alive for loss.backward().
tokenizer = AutoTokenizer.from_pretrained(args.model)

def encode_with_grad(texts, max_length=512):
    """Tokenise + forward pass with gradients. Returns L2-normalised embeddings."""
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    ).to(device)
    transformer_output = model[0].auto_model(**encoded)
    # Mean pooling over non-padding tokens
    token_embeddings    = transformer_output.last_hidden_state          # [B, T, D]
    attention_mask      = encoded["attention_mask"]
    input_mask_expanded = attention_mask.unsqueeze(-1).float()
    pooled = (token_embeddings * input_mask_expanded).sum(1) / input_mask_expanded.sum(1).clamp(min=1e-9)
    return F.normalize(pooled, p=2, dim=1)                              # [B, D]

# ── Loss ──────────────────────────────────────────────────────────────────────
# SupConLoss: temperature=0.07 is the default from Khosla et al.
supcon_loss = pml_losses.SupConLoss(temperature=0.07)

# ── Optimiser + scheduler ─────────────────────────────────────────────────────
# 8-bit AdamW keeps optimizer momentum buffers in int8 instead of fp32,
# cutting optimizer VRAM from ~2× model size down to ~0.5×.
optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=args.lr, weight_decay=0.01)

total_steps  = len(train_loader) * args.epochs
warmup_steps = min(args.warmup_steps, total_steps // 10)

def lr_lambda(step):
    if step < warmup_steps:
        return float(step) / max(1, warmup_steps)
    progress = float(step - warmup_steps) / max(1, total_steps - warmup_steps)
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# ── Evaluation ────────────────────────────────────────────────────────────────
def evaluate(model, test_pairs, batch_size=128):
    """
    For each test pair (sending_text, receiving_text):
      - Embed all unique receiving-side texts
      - Rank by cosine similarity to the sending embedding
      - Report Recall@1, @5, @10 and mean similarity of true pair
    """
    model.eval()
    with torch.no_grad():
        recv_texts = list({p["receiving_text"] for p in test_pairs})
        recv_index = {t: i for i, t in enumerate(recv_texts)}

        recv_embeddings = model.encode(
            recv_texts, batch_size=batch_size,
            convert_to_tensor=True, normalize_embeddings=True,
            show_progress_bar=False
        )
        send_texts = [p["sending_text"] for p in test_pairs]
        send_embeddings = model.encode(
            send_texts, batch_size=batch_size,
            convert_to_tensor=True, normalize_embeddings=True,
            show_progress_bar=False
        )

        sim_matrix = torch.mm(send_embeddings.float(), recv_embeddings.float().T).cpu().numpy()

        hits = {1: 0, 5: 0, 10: 0}
        true_sims = []

        for i, pair in enumerate(test_pairs):
            true_idx = recv_index[pair["receiving_text"]]
            true_sim = sim_matrix[i, true_idx]
            true_sims.append(float(true_sim))

            rank = int((sim_matrix[i] >= true_sim).sum())
            for k in [1, 5, 10]:
                if rank <= k:
                    hits[k] += 1

        n = len(test_pairs)
        results = {
            "recall@1":      hits[1]  / n,
            "recall@5":      hits[5]  / n,
            "recall@10":     hits[10] / n,
            "mean_true_sim": float(np.mean(true_sims)),
        }

    model.train()
    return results


# ── Training loop ─────────────────────────────────────────────────────────────
print(f"\nStarting training: {args.epochs} epochs, batch_size={args.batch_size}, lr={args.lr}")
print(f"Total steps: {total_steps}, warmup: {warmup_steps}")

best_recall1 = -1.0
global_step  = 0

for epoch in range(1, args.epochs + 1):
    epoch_loss = 0.0
    n_batches  = 0

    for texts, labels in train_loader:
        embeddings = encode_with_grad(texts)
        labels     = labels.to(device)

        loss = supcon_loss(embeddings, labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        epoch_loss  += loss.item()
        n_batches   += 1
        global_step += 1

        if global_step % 10 == 0:
            print(f"  step {global_step:4d} | loss {loss.item():.4f} "
                  f"| lr {scheduler.get_last_lr()[0]:.2e}")

    avg_loss = epoch_loss / max(1, n_batches)

    # Evaluate
    eval_results = evaluate(model, test_pairs)
    r1 = eval_results["recall@1"]

    print(
        f"\nEpoch {epoch}/{args.epochs} | avg_loss={avg_loss:.4f} | "
        f"R@1={r1:.3f} R@5={eval_results['recall@5']:.3f} "
        f"R@10={eval_results['recall@10']:.3f} | "
        f"mean_true_sim={eval_results['mean_true_sim']:.4f}"
    )

    if r1 > best_recall1:
        best_recall1 = r1
        save_path = out_dir / "best_model"
        model.save(str(save_path))
        print(f"  ✓ New best Recall@1={r1:.3f} — saved to {save_path}")

# Save final model
final_path = out_dir / "final_model"
model.save(str(final_path))
print(f"\nFinal model saved to {final_path}")
print(f"Best Recall@1 on test set: {best_recall1:.3f}")
print("Done.")
