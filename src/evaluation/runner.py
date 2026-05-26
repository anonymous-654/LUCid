from typing import Any, Dict

from src.llm_client import query_llm
from src.evaluation.prompts import prepare_mcq_prompt, prepare_suitability_prompt
from src.evaluation.utils import (
    extract_json_from_text,
    extract_mcq_references,
    get_evaluation_route,
    get_expected_category,
    get_letter_value,
    get_question,
)

EVALUATOR_MODEL = "gpt-5.4-mini"


def evaluate_entry(entry: Dict[str, Any], model_name: str = EVALUATOR_MODEL) -> Dict[str, Any]:
    question = get_question(entry)
    answer = entry.get("hypothesis", "")
    expected_category = get_expected_category(entry)
    dimension, prompt_type, prompt_subtype = get_evaluation_route(entry)

    if prompt_type in ("unknown", "", None):
        print("\n[ERROR] Unknown evaluation route detected")
        print(f"Dimension: {dimension}")
        print(f"Prompt type: {prompt_type}")
        print(f"Prompt subtype: {prompt_subtype}")
        print(f"Entry ID: {entry.get('query_id')}")
        print(f"Query dimension (raw): {entry.get('query_dimension')}")
        print(f"Query topic: {entry.get('query_topic')}")
        print(f"Full entry: {entry}\n")

        raise ValueError(
            f"Unsupported evaluation route: dimension={dimension}, "
            f"prompt_type={prompt_type}, prompt_subtype={prompt_subtype}")

    result = dict(entry)
    result.pop("haystack_sessions", None)
    result.pop("haystack_session_ids", None)
    result["evaluator_model"] = model_name
    result["evaluator_dimension"] = dimension

    if prompt_type == "mcq":
        references = extract_mcq_references(entry)
        prompt, label_letter, include_all, full_ref_list = prepare_mcq_prompt(
            question=question,
            answer=answer,
            references=references,
            expected_category=expected_category,
            dimension=dimension,
        )

        raw_eval, _, _ = query_llm(
            model_path=model_name,
            user_prompt=prompt,
            temperature=0,
            max_tokens=200,
            print_tokens=False,
        )
        parsed = extract_json_from_text(raw_eval)

        pred_letter = str(parsed.get("evaluation", "")).strip().upper()
        reasoning = str(parsed.get("reasoning", "")).strip()
        pred_value = get_letter_value(pred_letter, references, include_all)
        is_match = 1 if (label_letter is not None and pred_letter == label_letter) else 0

        result["evaluator_prompt_type"] = "mcq"
        result["evaluator_prompt_subtype"] = prompt_subtype
        result["evaluator_options"] = full_ref_list
        result["evaluator_expected"] = expected_category
        result["evaluator_expected_letter"] = label_letter
        result["evaluator_prediction"] = pred_value
        result["evaluator_prediction_letter"] = pred_letter
        result["evaluator_match"] = is_match
        result["evaluator_reasoning"] = reasoning
        result["evaluator_raw_output"] = raw_eval
        return result

    if prompt_type == "suitability":
        prompt = prepare_suitability_prompt(
            question=question,
            answer=answer,
            expected_category=expected_category,
            dimension=dimension,
            prompt_subtype=prompt_subtype,
        )

        raw_eval, _, _ = query_llm(
            model_path=model_name,
            user_prompt=prompt,
            temperature=0,
            max_tokens=200,
            print_tokens=False,
        )
        parsed = extract_json_from_text(raw_eval)

        pred = str(parsed.get("evaluation", "")).strip().lower()
        reasoning = str(parsed.get("reasoning", "")).strip()
        is_match = 1 if pred == "match" else 0

        result["evaluator_prompt_type"] = "suitability"
        result["evaluator_prompt_subtype"] = prompt_subtype
        result["evaluator_expected"] = expected_category
        result["evaluator_prediction"] = pred
        result["evaluator_match"] = is_match
        result["evaluator_reasoning"] = reasoning
        result["evaluator_raw_output"] = raw_eval
        return result

    result["evaluator_prompt_type"] = "unknown"
    result["evaluator_prompt_subtype"] = prompt_subtype
    result["evaluator_expected"] = expected_category
    result["evaluator_prediction"] = None
    result["evaluator_match"] = 0
    result["evaluator_reasoning"] = (
        f"Unsupported evaluation route: dimension={dimension}, prompt_type={prompt_type}, prompt_subtype={prompt_subtype}"
    )
    return result