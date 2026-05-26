#!/usr/bin/env bash

#SBATCH --mail-user=cokite@umich.edu
#SBATCH --mail-type=ALL
#SBATCH --time=90:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --account=mihalcea98

set -euo pipefail

module load python/3.11.5
source .venv/bin/activate

export PYTHONIOENCODING=utf-8

cd /gpfs/accounts/chaijy_root/chaijy2/cokite/LatentGrounding
python -m src.retrieval.index_expansion.batch_expansion_session_userfact