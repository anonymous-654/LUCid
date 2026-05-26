#!/usr/bin/env bash

# SBATCH --partition=spgpu
#SBATCH --time=60:10:00
#SBATCH --cpus-per-task=2
# SBATCH --gpus=1
#SBATCH --mem=32G



# Load modules
module load python/3.11.5 cuda
source .venv/bin/activate 
# BAAI/bge-reranker-v2-m3 BAAI/bge-reranker-v2-gemma Qwen/Qwen3-Reranker-0.6B, Qwen/Qwen3-Reranker-8B
# gemini-3-flash-preview, Qwen/Qwen3.5-27B-FP8, claude-3-haiku-20240307, gpt-5.4-mini

in_file="data/lucid_c.json"
out_dir="src/reranker/reranker_logs"
retriever="oracle-session"
granularity="session"
reranker="claude-haiku-4-5-20251001"

mkdir -p $out_dir

# python -m src.reranker.run_reranker \
#   --in_file $in_file \
#   --out_dir $out_dir \
#   --granularity $granularity \
#   --reranker $reranker \
#   --top_k 20 \
#   --clamp_threshold \
#   --promptreason

# python -m src.reranker.aggregate_results \
#   --input_dir src/reranker/reranker_logs/ \
#   --pattern "*.jsonl" \
#   --output_file src/reranker/reranker_logs/summary.csv \
#   # --promptreason false
  
python -m src.reranker.aggregate_reranker_logs \
  --logs_root src/reranker/reranker_logs \
  --out_file src/reranker/reranker_logs/aggregated_results.json
