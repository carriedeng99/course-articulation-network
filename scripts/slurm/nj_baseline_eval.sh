#!/bin/bash
#SBATCH --job-name=nj_baseline
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/users/rdeng5/Desktop/NJ/logs/nj_baseline_%j.out
#SBATCH --error=/users/rdeng5/Desktop/NJ/logs/nj_baseline_%j.err

mkdir -p /users/rdeng5/Desktop/NJ/logs

export PATH="/users/rdeng5/.conda/envs/transfer_net/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python /users/rdeng5/Desktop/NJ/nj_baseline_eval.py \
    --test  /users/rdeng5/Desktop/NJ/nj_ft_data/nj_test.jsonl \
    --model Qwen/Qwen3-Embedding-8B \
    --batch_size 128
