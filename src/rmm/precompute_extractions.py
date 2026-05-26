from __future__ import annotations

import argparse
import hashlib
import json
import os
from typing import Any, Dict

from tqdm import tqdm

from src.rmm.llm import LLMClient
from src.rmm.memory import extract_session_memories


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


def make_session_content_hash(session_turns) -> str:
    normalized = json.dumps(session_turns, ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()


def make_session_key(sess_id: str, session_turns) -> str:
    content_hash = make_session_content_hash(session_turns)
    return f"{sess_id}__{content_hash[:12]}"


def collect_unique_sessions(in_data) -> Dict[str, Any]:
    unique_sessions: Dict[str, Any] = {}

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
    out_dir = os.path.dirname(args.out_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

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

    session_items = list(unique_sessions.items())

    for i, (session_key, payload) in enumerate(
        tqdm(session_items, desc="Precomputing extractions")
    ):
        if session_key in precomputed:
            continue

        sess_id = payload["session_id"]
        session_turns = payload["session_turns"]

        extracted, dialogue_turns = extract_session_memories(llm, session_turns)

        precomputed[session_key] = {
            "session_key": session_key,
            "session_id": sess_id,
            "content_hash": payload["content_hash"],
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