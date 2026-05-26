#!/usr/bin/env bash

#SBATCH --mail-type=ALL
#SBATCH --partition=spgpu
#SBATCH --time=120:00:00
#SBATCH --cpus-per-task=2
#SBATCH --gpus=1
#SBATCH --mem=32G
# SBATCH --array=0-5

module load python/3.11.5 cuda
source .venv/bin/activate

# mkdir -p src/rmm/logs

# python -m src.rmm.precompute_extractions \
#   --in_file data/lucid_b.json \
#   --out_file src/rmm/logs/precomputed_extractions.json \
#   --model Qwen/Qwen3.5-27B-FP8 \
#   --base_url http://gl1500.arc-ts.umich.edu:8002/v1 \
#   # --limit 2

# python -m src.rmm.run_rmm_retrieval \
#   --in_file data/lucid_b.json \
#   --out_dir src/rmm/logs \
#   --model Qwen/Qwen3.5-27B-FP8 \
#   --base_url http://gl1500.arc-ts.umich.edu:8002/v1 \
#   --retriever flat-contriever \
#   --memory_top_k 5 \
#   --precomputed_extractions_file src/rmm/logs/precomputed_extractions.json \
#   # --limit 2

# NUM_SHARDS=6
# SHARD_ID=${SLURM_ARRAY_TASK_ID}

# python -m src.rmm.run_rmm_retrieval \
#   --in_file data/lucid_s.json \
#   --out_dir src/rmm/logs/shards \
#   --outfile_prefix lucid_s \
#   --model Qwen/Qwen3.5-27B-FP8 \
#   --base_url http://gl1500.arc-ts.umich.edu:8002/v1 \
#   --retriever flat-contriever \
#   --memory_top_k 5 \
#   --precomputed_extractions_file src/rmm/logs/precomputed_extractions.json \
#   --shard_id "${SHARD_ID}" \
#   --num_shards "${NUM_SHARDS}" \
#   # --limit 8

# python -m src.rmm.aggregate_rmm_shards \
#   --input_glob "src/rmm/logs/shards/*_shard*of006.jsonl" \
#   --merged_jsonl src/rmm/logs/all_shards_merged.jsonl \
#   --summary_json src/rmm/logs/all_shards_metrics.json

#  python -m src.rmm.aggregate_rmm_shards

python -m src.rmm.run_rmm_reranker \
    --in_file "src/rmm/logs/all_shards_merged.jsonl" \
    --out_file "src/rmm/logs/all_shards_merged_rerank.jsonl" \
    --reranker_type "bge-gemma" \
    --reranker_model "BAAI/bge-reranker-v2-gemma"
