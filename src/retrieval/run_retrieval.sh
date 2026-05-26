#!/usr/bin/env bash

#SBATCH --partition=spgpu
#SBATCH --time=60:10:00
#SBATCH --cpus-per-task=2
#SBATCH --gpus=1
#SBATCH --mem=32G

set -euo pipefail

export PYTHONIOENCODING=utf-8

# Load modules
module load python/3.11.5 cuda
source .venv/bin/activate


# ['flat-bm25', 'flat-contriever', 'flat-stella', 'flat-gte', 'oracle']:

# src/retrieval/index_expansion/index_expansion_logs/lucid_b.json.session-userfact.ICL.json

in_file="data/lucid_s.json"
retriever="flat-contriever"
granularity="turn"
index_expansion_result_join_mode="none"
index_expansion_result_cache="none"
cache_dir="none"
outfile_prefix="none"

home_dir=$(pwd)

if [[ "${index_expansion_result_cache}" == "none" ]]; then
    out_dir="${home_dir}/src/retrieval/retrieval_logs/${retriever}/${granularity}"
else
    out_dir="${home_dir}/src/retrieval/retrieval_logs/${retriever}_expansion_w_session-userfact/joinmode_${index_expansion_result_join_mode}/${granularity}"
fi

mkdir -p "${out_dir}"

cmd=(
    python -m src.retrieval.run_retrieval
    --in_file "${in_file}"
    --out_dir "${out_dir}"
    --retriever "${retriever}"
    --granularity "${granularity}"
    --cache_dir "${cache_dir}"
)

if [[ "${outfile_prefix}" != "none" ]]; then
    cmd+=(--outfile_prefix "${outfile_prefix}")
fi

if [[ "${index_expansion_result_cache}" != "none" ]]; then
    cmd+=(--index_expansion_result_cache "${index_expansion_result_cache}")
    cmd+=(--index_expansion_result_join_mode "${index_expansion_result_join_mode}")
fi

printf 'Running command:\n%s\n' "${cmd[*]}"
"${cmd[@]}"
