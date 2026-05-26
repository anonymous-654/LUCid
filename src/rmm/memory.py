from __future__ import annotations

import json
import re
import uuid
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from src.rmm.prompts import EXTRACTION_SYSTEM_PROMPT, UPDATE_SYSTEM_PROMPT
from src.rmm.utils import make_session_key


def _strip_code_fences(text: str) -> str:
    return text.replace("```json", "").replace("```", "").strip()


def _parse_extraction_output(text: str) -> List[Dict[str, Any]]:
    """
    Allowed outputs:
      - JSON object with top-level key "extracted_memories"
      - literal NO_TRAIT
    """
    text = _strip_code_fences(text).strip()

    if text == "NO_TRAIT":
        return []

    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not m:
            raise ValueError(f"Could not parse extraction output: {text}")
        obj = json.loads(m.group(0))

    extracted = obj.get("extracted_memories", [])
    if not isinstance(extracted, list):
        return []

    cleaned: List[Dict[str, Any]] = []
    for item in extracted:
        if not isinstance(item, dict):
            continue

        summary = str(item.get("summary", "")).strip()
        refs_raw = item.get("reference", [])

        refs: List[int] = []
        if isinstance(refs_raw, list):
            for x in refs_raw:
                if isinstance(x, int):
                    refs.append(x)
                elif isinstance(x, str) and x.strip().isdigit():
                    refs.append(int(x.strip()))

        if summary:
            cleaned.append(
                {
                    "summary": summary,
                    "reference": refs,
                }
            )

    return cleaned


def _parse_update_actions(text: str) -> List[Tuple[str, Optional[int], Optional[str]]]:
    """
    Allowed actions:
      Add()
      Merge(index, merged_summary)
    """
    text = _strip_code_fences(text).strip()
    if not text:
        raise ValueError("Empty update action output")

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    actions: List[Tuple[str, Optional[int], Optional[str]]] = []

    for line in lines:
        if line == "Add()":
            actions.append(("add", None, None))
            continue

        m = re.match(r"^Merge\(\s*(\d+)\s*,\s*(.*)\)$", line, flags=re.DOTALL)
        if m:
            idx = int(m.group(1))
            merged_summary = m.group(2).strip()
            actions.append(("merge", idx, merged_summary))
            continue

        raise ValueError(f"Could not parse action line: {line}")

    return actions


def _choose_single_action(
    actions: List[Tuple[str, Optional[int], Optional[str]]],
) -> Tuple[str, Optional[int], Optional[str]]:
    if not actions:
        return "add", None, None
    return actions[0]


def build_dialogue_turns(session_turns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert raw session messages into grouped dialogue turns.

    Input:
      [{"role": "user"|"assistant", "content": "..."}]

    Output:
      [
        {
          "turn_id": 0,
          "speaker_1": "...",
          "speaker_2": "...",
          "raw_items": [...]
        },
        ...
      ]
    """
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


def format_dialogue_turns_for_extraction(dialogue_turns: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for turn in dialogue_turns:
        lines.append(f"Turn {turn['turn_id']}:")
        lines.append(f"- SPEAKER_1: {turn['speaker_1']}")
        lines.append(f"- SPEAKER_2: {turn['speaker_2']}")
    return "\n".join(lines)


def format_session_for_extraction(session_turns: List[Dict[str, Any]]) -> str:
    dialogue_turns = build_dialogue_turns(session_turns)
    return format_dialogue_turns_for_extraction(dialogue_turns)


def extract_session_memories(
    llm,
    session_turns: List[Dict[str, Any]],
    max_tokens: int = 500,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Returns:
      extracted_memories, dialogue_turns
    """
    dialogue_turns = build_dialogue_turns(session_turns)
    session_text = format_dialogue_turns_for_extraction(dialogue_turns)
    prompt = EXTRACTION_SYSTEM_PROMPT.format(session_text)

    messages = [{"role": "user", "content": prompt}]
    raw = llm.chat(messages, max_tokens=max_tokens, temperature=0.0)
    extracted = _parse_extraction_output(raw)

    return extracted, dialogue_turns


def build_reference_payload(
    session_id: str,
    dialogue_turns: List[Dict[str, Any]],
    turn_ids: List[int],
) -> Dict[str, Any]:
    safe_turn_ids = [
        i for i in turn_ids
        if isinstance(i, int) and 0 <= i < len(dialogue_turns)
    ]

    referenced_turns = [dialogue_turns[i] for i in safe_turn_ids]

    return {
        "session_id": session_id,
        "turn_ids": safe_turn_ids,
        "raw_turns": [
            {
                "turn_id": t["turn_id"],
                "speaker_1": t["speaker_1"],
                "speaker_2": t["speaker_2"],
            }
            for t in referenced_turns
        ],
    }


def _append_new_memory(
    memory_bank: List[Dict[str, Any]],
    memory_bank_embeddings: List[np.ndarray],
    new_memory: Dict[str, Any],
    new_memory_embedding: np.ndarray,
    session_id: str,
    dialogue_turns: List[Dict[str, Any]],
) -> None:
    memory_bank.append(
        {
            "memory_id": str(uuid.uuid4()),
            "summary": new_memory["summary"],
            "source_session_ids": [session_id],
            "references": [
                build_reference_payload(session_id, dialogue_turns, new_memory["reference"])
            ],
        }
    )
    memory_bank_embeddings.append(new_memory_embedding.astype(np.float32))


def update_memory_bank(
    llm,
    embedder,
    memory_bank: List[Dict[str, Any]],
    memory_bank_embeddings: List[np.ndarray],
    new_memory: Dict[str, Any],
    session_id: str,
    dialogue_turns: List[Dict[str, Any]],
    memory_top_k: int,
) -> Tuple[List[Dict[str, Any]], List[np.ndarray]]:
    new_embedding = embedder.encode_one(new_memory["summary"]).astype(np.float32)

    if not memory_bank:
        _append_new_memory(
            memory_bank=memory_bank,
            memory_bank_embeddings=memory_bank_embeddings,
            new_memory=new_memory,
            new_memory_embedding=new_embedding,
            session_id=session_id,
            dialogue_turns=dialogue_turns,
        )
        return memory_bank, memory_bank_embeddings

    bank_embeddings_np = np.vstack(memory_bank_embeddings)
    top_idx = embedder.top_k_from_precomputed(
        query_embedding=new_embedding,
        candidate_embeddings=bank_embeddings_np,
        k=memory_top_k,
    )
    history_summaries = [memory_bank[i]["summary"] for i in top_idx]

    history_json = {"history_summaries": history_summaries}
    new_json = {"new_summary": new_memory["summary"]}

    prompt = UPDATE_SYSTEM_PROMPT.format(
        json.dumps(history_json, ensure_ascii=False),
        json.dumps(new_json, ensure_ascii=False),
    )

    messages = [{"role": "user", "content": prompt}]

    try:
        action_text = llm.chat(messages, max_tokens=120, temperature=0.0)
        actions = _parse_update_actions(action_text)
        action, local_idx, merged_summary = _choose_single_action(actions)
    except Exception:
        action, local_idx, merged_summary = "add", None, None

    if action == "add" or not top_idx:
        _append_new_memory(
            memory_bank=memory_bank,
            memory_bank_embeddings=memory_bank_embeddings,
            new_memory=new_memory,
            new_memory_embedding=new_embedding,
            session_id=session_id,
            dialogue_turns=dialogue_turns,
        )
        return memory_bank, memory_bank_embeddings

    if local_idx is None or local_idx < 0 or local_idx >= len(top_idx):
        _append_new_memory(
            memory_bank=memory_bank,
            memory_bank_embeddings=memory_bank_embeddings,
            new_memory=new_memory,
            new_memory_embedding=new_embedding,
            session_id=session_id,
            dialogue_turns=dialogue_turns,
        )
        return memory_bank, memory_bank_embeddings

    bank_idx = top_idx[local_idx]
    final_summary = merged_summary.strip() if merged_summary else new_memory["summary"]

    memory_bank[bank_idx]["summary"] = final_summary
    if session_id not in memory_bank[bank_idx]["source_session_ids"]:
        memory_bank[bank_idx]["source_session_ids"].append(session_id)

    memory_bank[bank_idx]["references"].append(
        build_reference_payload(session_id, dialogue_turns, new_memory["reference"])
    )

    memory_bank_embeddings[bank_idx] = embedder.encode_one(final_summary).astype(np.float32)
    return memory_bank, memory_bank_embeddings


def reflect_over_sessions(
    entry,
    llm,
    embedder,
    precomputed_extractions: Dict[str, Any],
    memory_top_k: int,
):
    """
    Build an entry-local memory bank using precomputed session-level extractions.

    precomputed_extractions must be keyed by the exact sess_id used in the entry:
      {
        sess_id: {
          "session_id": sess_id,
          "dialogue_turns": [...],
          "extracted_memories": [...] or "NO_TRAIT"
        }
      }
    """
    memory_bank: List[Dict[str, Any]] = []
    memory_bank_embeddings: List[np.ndarray] = []
    extraction_log = []

    for sess_id, session_turns in zip(
        entry["haystack_session_ids"],
        entry["haystack_sessions"],
    ):
        session_key = make_session_key(sess_id, session_turns)

        if session_key not in precomputed_extractions:
            raise KeyError(
                f"Session key {session_key} not found for sess_id={sess_id}"
            )

        payload = precomputed_extractions[session_key]
        # if sess_id not in precomputed_extractions:
        #     raise KeyError(
        #         f"Session id {sess_id} not found in precomputed extractions. "
        #         f"Make sure preprocessing covered all sessions."
        #     )

        # payload = precomputed_extractions[sess_id]
        dialogue_turns = payload["dialogue_turns"]
        cached_extracted = payload["extracted_memories"]

        extracted = [] if cached_extracted == "NO_TRAIT" else cached_extracted

        extraction_log.append(
            {
                "session_id": sess_id,
                "dialogue_turns": [
                    {
                        "turn_id": t["turn_id"],
                        "speaker_1": t["speaker_1"],
                        "speaker_2": t["speaker_2"],
                    }
                    for t in dialogue_turns
                ],
                "extracted_memories": extracted,
            }
        )

        for mem in extracted:
            memory_bank, memory_bank_embeddings = update_memory_bank(
                llm=llm,
                embedder=embedder,
                memory_bank=memory_bank,
                memory_bank_embeddings=memory_bank_embeddings,
                new_memory=mem,
                session_id=sess_id,
                dialogue_turns=dialogue_turns,
                memory_top_k=memory_top_k,
            )

    return memory_bank, memory_bank_embeddings, extraction_log


def retrieve_over_memory_bank(
    query,
    memory_bank,
    retriever_master,
    retriever,
    cached_doc_vectors: Optional[torch.Tensor] = None,
):
    if not memory_bank:
        return []

    if retriever == "oracle":
        correct_idx = [
            i for i, m in enumerate(memory_bank)
            if any("answer" in sid for sid in m["source_session_ids"])
        ]
        incorrect_idx = [
            i for i, m in enumerate(memory_bank)
            if not any("answer" in sid for sid in m["source_session_ids"])
        ]
        return correct_idx + incorrect_idx

    if cached_doc_vectors is None:
        corpus = [m["summary"] for m in memory_bank]
        cached_doc_vectors = retriever_master.encode_texts(corpus)

    return retriever_master.rank_with_cached_docs(query, cached_doc_vectors)