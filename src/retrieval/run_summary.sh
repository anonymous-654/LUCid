#!/usr/bin/env bash

#SBATCH --time=60:10:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G



# Load modules
module load python/3.11.5 cuda
source .venv/bin/activate 


# python -m src.retrieval.summarize_retrieval_logs

python -m src.retrieval.aggregate_retrieval_logs \
  --logs_root src/retrieval/retrieval_logs \
  --out_file src/retrieval/retrieval_logs/aggregated_results.json
  