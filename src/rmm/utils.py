from __future__ import annotations

import hashlib
import json
import os
import re
from typing import Any, Dict, List


def get_outfile_prefix(args):
    if args.outfile_prefix is not None and args.outfile_prefix.lower() != "none":
        return args.outfile_prefix
    return os.path.basename(args.in_file)


def strip_code_fences(text: str) -> str:
    return text.replace("```json", "").replace("```", "").strip()


def parse_extraction_json(text: str):
    text = strip_code_fences(text)
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not m:
            raise
        obj = json.loads(m.group(0))
    return obj.get("extracted_memories", [])


def parse_update_action(text: str):
    text = text.strip()

    if text == "Add()":
        return "add", None, None

    m = re.match(r"^Merge\(\s*(\d+)\s*,\s*(.*)\)$", text, flags=re.DOTALL)
    if m:
        idx = int(m.group(1))
        merged_summary = m.group(2).strip()
        return "merge", idx, merged_summary

    raise ValueError(f"Could not parse action: {text}")


def build_dialogue_turns(session_turns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    dialogue_turns: List[Dict[str, Any]] = []
    i = 0
    turn_id = 0
    n = len(session_turns)

    while i < n:
        speaker_1 = ""
        speaker_2 = ""
        raw_items: List[Dict[str, Any]] = []

        if i < n and session_turns[i].get("role") == "user":
            speaker_1 = str(session_turns[i].get("content", "")).strip()
            raw_items.append(session_turns[i])
            i += 1

        if i < n and session_turns[i].get("role") == "assistant":
            speaker_2 = str(session_turns[i].get("content", "")).strip()
            raw_items.append(session_turns[i])
            i += 1

        if not raw_items and i < n:
            speaker_2 = str(session_turns[i].get("content", "")).strip()
            raw_items.append(session_turns[i])
            i += 1

        dialogue_turns.append(
            {
                "turn_id": turn_id,
                "speaker_1": speaker_1,
                "speaker_2": speaker_2,
                "raw_items": raw_items,
            }
        )
        turn_id += 1

    return dialogue_turns

def make_session_content_hash(session_turns):
    normalized = json.dumps(session_turns, ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()

def make_session_key(sess_id, session_turns):
    content_hash = make_session_content_hash(session_turns)
    return f"{sess_id}__{content_hash[:12]}"
