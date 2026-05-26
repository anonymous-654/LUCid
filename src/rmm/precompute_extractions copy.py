from __future__ import annotations

import argparse
import json
import hashlib
import os
from typing import Any, Dict

from tqdm import tqdm

from src.rmm.llm import LLMClient
from src.rmm.memory import extract_session_memories

from src.rmm.utils import make_session_content_hash, make_session_key


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--in_file", type=str, required=True)
    parser.add_argument("--out_file", type=str, required=True)

    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--base_url", type=str, required=True)
    parser.add_argument("--api_key", type=str, default="EMPTY")

    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--save_every", type=int, default=100)

    return parser.parse_args()


def collect_unique_sessions(in_data):
    unique_sessions = {}

    for entry in in_data:
        for sess_id, session_turns in zip(
            entry["haystack_session_ids"],
            entry["haystack_sessions"],
        ):
            session_key = make_session_key(sess_id, session_turns)

            if session_key not in unique_sessions:
                unique_sessions[session_key] = {
                    "session_id": sess_id,
                    "session_turns": session_turns,
                    "content_hash": make_session_content_hash(session_turns),
                }

    return unique_sessions


def main(args):
    os.makedirs(os.path.dirname(args.out_file) or ".", exist_ok=True)

    with open(args.in_file, "r") as f:
        in_data = json.load(f)

    if args.limit is not None:
        in_data = in_data[:args.limit]

    unique_sessions = collect_unique_sessions(in_data)
    print(f"Collected {len(unique_sessions)} unique sessions")

    precomputed = {}
    if os.path.isfile(args.out_file):
        with open(args.out_file, "r") as f:
            precomputed = json.load(f)
        print(f"Loaded existing precomputed extraction file from {args.out_file}")
        print(f"Already have {len(precomputed)} sessions cached")

    llm = LLMClient(
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
    )

    session_ids = list(unique_sessions.keys())

    for i, sess_id in enumerate(tqdm(session_ids, desc="Precomputing extractions")):
        if sess_id in precomputed:
            continue

        session_turns = unique_sessions[sess_id]
        extracted, dialogue_turns = extract_session_memories(llm, session_turns)

        precomputed[sess_id] = {
            "session_id": sess_id,
            "dialogue_turns": [
                {
                    "turn_id": t["turn_id"],
                    "speaker_1": t["speaker_1"],
                    "speaker_2": t["speaker_2"],
                }
                for t in dialogue_turns
            ],
            "extracted_memories": extracted if extracted else "NO_TRAIT",
        }

        if (i + 1) % args.save_every == 0:
            with open(args.out_file, "w") as f:
                json.dump(precomputed, f, ensure_ascii=False, indent=2)

    with open(args.out_file, "w") as f:
        json.dump(precomputed, f, ensure_ascii=False, indent=2)

    print(f"Saved precomputed extractions to {args.out_file}")
    print(f"Final number of sessions stored: {len(precomputed)}")


if __name__ == "__main__":
    args = parse_args()
    main(args)