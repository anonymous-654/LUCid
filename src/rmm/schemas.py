from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Dict, Any


@dataclass
class Turn:
    turn_id: int
    speaker: str
    text: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExtractedMemory:
    summary: str
    reference: List[int]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MemoryEntry:
    memory_id: str
    summary: str
    references: List[Dict[str, Any]]
    source_session_id: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)