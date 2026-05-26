#!/usr/bin/env bash
#SBATCH --mail-user=cokite@umich.edu
#SBATCH --mail-type=ALL
#SBATCH --partition=standard
#SBATCH --time=120:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --account=chaijy2
#SBATCH --array=0-7

set -euo pipefail

module load python/3.11.5
source .venv/bin/activate

NUM_SHARDS=8
SHARD_ID=${SLURM_ARRAY_TASK_ID}

python -m src.rmm.run_rmm_retrieval \
  --in_file data/lucid_b.json \
  --out_dir src/rmm/logs/shards \
  --outfile_prefix lucid_b \
  --model Qwen/Qwen3.5-27B-FP8 \
  --base_url http://gl1500.arc-ts.umich.edu:8002/v1 \
  --retriever flat-contriever \
  --memory_top_k 5 \
  --precomputed_extractions_file src/rmm/logs/precomputed_extractions.json \
  --shard_id "${SHARD_ID}" \
  --num_shards "${NUM_SHARDS}" \
  --limit 8
