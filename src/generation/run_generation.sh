#!/usr/bin/env bash

#SBATCH --time=60:10:00
#SBATCH --cpus-per-task=2 
#SBATCH --mem=32G



# Load modules
module load python/3.11.5 cuda
source .venv/bin/activate 

# gemini-3-flash-preview, Qwen/Qwen3.5-27B-FP8, claude-3-haiku-20240307, gpt-5.4-mini, claude-haiku-4-5
# ["orig-session", "flat-session", "oracle-session", "orig-turn", "flat-turn", "oracle-turn"]: ["no-retrieval", "gold"]


# the next to run is gold for chatgpt

# no-retrieval
in_file="data/lucid_s.json"
# in_file="src/retrieval/retrieval_logs/flat-contriever/session/lucid_s.json_retrievallog_session_flat-contriever_1953"
out_dir="src/generation/generation_logs/"
retriever="no-retrieval"
granularity="session"
model_name="claude-haiku-4-5"

mkdir -p $out_dir

python -m src.generation.generation \
    --in_file $in_file \
    --out_dir $out_dir \
    --model_name $model_name \
    --retriever_type $retriever \
    --topk_context 999 \
    --history_format "json" \
    --useronly true \
    --gen_length 1200
