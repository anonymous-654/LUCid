import json
import string
from typing import Any, Dict, List, Tuple


MCQ_DIMENSIONS = {"age_group", "domain", "location/country", "style_pref"}
SUITABILITY_DIMENSIONS = {"religion", "medical_health_condition"}

AGE_GROUP_SUITABILITY_TOPICS = {
    "drinking/explicit content",
    "entertainment",
}


def normalize_dimension(dim: str) -> str:
    dim = (dim or "").strip().lower()
    if dim == "country/location":
        return "location/country"
    if dim == "medical/health":
        return "medical_health_condition"
    return dim


def normalize_dimension(value: str) -> str:
    return str(value).strip().lower()


def normalize_topic(value: str) -> str:
    return str(value).strip().lower()


def get_query_dimension(entry: Dict[str, Any]) -> str:
    return normalize_dimension(entry.get("query_dimension", ""))


def get_query_topic(entry: Dict[str, Any]) -> str:
    return normalize_topic(entry.get("query_topic", ""))


def get_evaluation_route(entry: Dict[str, Any]) -> Tuple[str, str, str]:
    """
    Returns:
        (effective_dimension, prompt_type, prompt_subtype)
    """
    dimension = get_query_dimension(entry)
    topic = get_query_topic(entry)

    if dimension == "age_group" and topic == "drinking/explicit content":
        return dimension, "suitability", "teen_age_group_explicit"
    
    if dimension == "age_group" and topic == "entertainment":
        return dimension, "suitability", "teen_age_group_movie"
    
    if dimension == "age_group":
        return dimension, "suitability", "teen_age_group_other"

    if dimension in MCQ_DIMENSIONS:
        return dimension, "mcq", "default_mcq"

    if dimension == "religion":
        return dimension, "suitability", "religion"

    if dimension == "medical_health_condition":
        return dimension, "suitability", "medical_health_condition"

    return dimension, "unknown", "unknown"


def safe_load_json_or_jsonl(path: str):
    try:
        return json.load(open(path))
    except Exception:
        return [json.loads(line) for line in open(path).readlines()]


def get_question(entry: Dict[str, Any]) -> str:
    return entry.get("query", entry.get("question", ""))


def get_expected_category(entry: Dict[str, Any]) -> str:
    return entry.get("expected_category", entry.get("expected_domain", ""))


def get_query_dimension(entry: Dict[str, Any]) -> str:
    return normalize_dimension(entry.get("query_dimension", ""))


def extract_answer_variations(entry: Dict[str, Any], expected_category: str) -> List[str]:
    values = entry.get("answer_variations", [])
    if not isinstance(values, list):
        values = []

    cleaned = []
    seen = set()

    for v in values:
        if not isinstance(v, str):
            continue
        vv = v.strip()
        if vv and vv not in seen:
            cleaned.append(vv)
            seen.add(vv)

    if expected_category and expected_category not in seen:
        cleaned.append(expected_category)

    return cleaned


def extract_mcq_references(entry: Dict[str, Any]) -> List[str]:
    expected_category = get_expected_category(entry)
    return extract_answer_variations(entry, expected_category)


def build_options(references: List[str], include_all_of_the_above: bool) -> Tuple[str, List[str], List[str]]:
    full_refs = references.copy()
    if include_all_of_the_above:
        full_refs.append("All of the above")
    full_refs.append("None of the above")

    options = list(string.ascii_uppercase)[: len(full_refs)]
    ref_text = ""
    for i, option in enumerate(full_refs):
        ref_text += f"\n{options[i]}. {full_refs[i]}"
    return ref_text, options, full_refs


def get_letter_value(letter: str, references: List[str], include_all_of_the_above: bool):
    _, options, full_refs = build_options(references, include_all_of_the_above)
    letter = (letter or "").strip().upper()
    if letter in options:
        return full_refs[options.index(letter)]
    return None


def extract_json_from_text(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    try:
        return json.loads(text)
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except Exception:
            pass

    return {}