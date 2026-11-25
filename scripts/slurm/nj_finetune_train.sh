#!/bin/bash
#SBATCH --job-name=nj_finetune
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/users/rdeng5/Desktop/NJ/logs/nj_finetune_%j.out
#SBATCH --error=/users/rdeng5/Desktop/NJ/logs/nj_finetune_%j.err

mkdir -p /users/rdeng5/Desktop/NJ/logs

export PATH="/users/rdeng5/.conda/envs/transfer_net/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python /users/rdeng5/Desktop/NJ/nj_finetune_train.py \
    --train /users/rdeng5/Desktop/NJ/nj_ft_data/nj_train.jsonl \
    --test  /users/rdeng5/Desktop/NJ/nj_ft_data/nj_test.jsonl \
    --model Qwen/Qwen3-Embedding-8B \
    --out_dir /users/rdeng5/Desktop/NJ/nj_finetuned \
    --epochs 5 \
    --batch_size 16 \
    --n_multi_comp 16 \
    --lr 2e-5 \
    --seed 42
