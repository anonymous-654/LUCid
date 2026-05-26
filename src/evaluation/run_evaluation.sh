#!/usr/bin/env bash

#SBATCH --mail-type=ALL
#SBATCH --time=60:10:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G

module load python/3.11.5 cuda
source .venv/bin/activate

out_dir="src/evaluation/evaluation_logs/"

# in_file="input file to evaluate"


mkdir -p "$out_dir"

python -m src.evaluation.evaluation \
    --in_file "$in_file" \
    --out_dir "$out_dir"