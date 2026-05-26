import re
import json
import copy
import os
from tqdm import tqdm
import argparse
import tiktoken

from src.llm_client import query_llm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_file", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--out_file_suffix", type=str, default="")

    parser.add_argument("--model_name", type=str, required=True)

    parser.add_argument("--retriever_type", type=str, required=True)
    parser.add_argument("--topk_context", type=int, required=True)
    parser.add_argument("--history_format", type=str, required=True, choices=["json", "nl"])
    parser.add_argument("--useronly", type=str, required=True, choices=["true", "false"])

    parser.add_argument("--gen_length", type=int, default=800)

    return parser.parse_args()


def check_args(args):
    print(args, flush=True)


def truncate_with_tiktoken(text: str, tokenizer, max_length: int) -> str:
    tokens = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    if len(tokens) > max_length:
        print(f"Truncating from {len(tokens)} to {max_length}", flush=True)
        text = tokenizer.decode(tokens[:max_length])
    return text

def normalize_speakers(text):
    if not text:
        return ""
    text = re.sub(r"\bSPEAKER[_ ]?1\b", "User", text, flags=re.IGNORECASE)
    text = re.sub(r"\bSPEAKER[_ ]?2\b", "Assistant", text, flags=re.IGNORECASE)
    return text


def build_rmm_memory_json(ret_result_entry, memory_idx, useronly=False):
    summary = normalize_speakers(ret_result_entry.get("text", "").strip())

    messages = []

    # system message = summary
    if summary:
        messages.append({
            "role": "system",
            "content": f"Memory [{memory_idx}]: {summary}"
        })

    # conversation turns
    for ref in ret_result_entry.get("references", []):
        for turn in ref.get("raw_turns", []):
            user_text = turn.get("speaker_1")
            assistant_text = turn.get("speaker_2")

            if user_text:
                messages.append({
                    "role": "user",
                    "content": user_text
                })

            if assistant_text and not useronly:
                messages.append({
                    "role": "assistant",
                    "content": assistant_text
                })

    return messages

def format_gold_dimension(query_dimension: str, expected_category: str) -> str:
    dim = query_dimension.strip().lower()
    value = expected_category.strip()

    if dim in {"age_group", "religion"}:
        return f"User feature: user is {value}."
    elif dim == "domain":
        return f"User feature: user domain is {value}."
    elif dim == "style_pref":
        return f"User feature: user mostly prefers {value} responses."
    elif dim in {"location/country", "location", "country"}:
        return f"User feature: user location/country is {value}."
    elif dim in {"medical_health_condition", "health", "medical"}:
        return f"User feature: user mentions {value}."
    else:
        return f"Relevant user information: {value}."


def prepare_prompt(
    entry,
    retriever_type,
    topk_context: int,
    useronly: bool,
    history_format: str,
    tokenizer,
    max_retrieval_length: int,
):
    if retriever_type == "no-retrieval":
        answer_prompt_template = "{}"
    elif retriever_type == "gold":
        answer_prompt_template = "Question: {}. {}\nAnswer:"
    else:
        answer_prompt_template = (
            "Please answer the question.\n\n"
            "History Chats:\n\n{}\n"
            "Question: {}\n"
            "Answer:"
        )

    question_string = entry["query"]
    expected_category = entry["expected_category"]
    query_dimension = entry["query_dimension"]

    corpusid2entry = {}
    for session_id, session_entry in zip(entry["haystack_session_ids"], entry["haystack_sessions"]):
        corpusid2entry[session_id] = session_entry
        for i_turn, turn_entry in enumerate(session_entry):
            corpusid2entry[f"{session_id}_{i_turn+1}"] = turn_entry

    retrieved_chunks = []

    if retriever_type == "orig-session":
        for session_entry in entry["haystack_sessions"]:
            if useronly:
                retrieved_chunks.append([x for x in session_entry if x["role"] == "user"])
            else:
                retrieved_chunks.append(session_entry)

    elif retriever_type == "orig-turn":
        for session_entry in entry["haystack_sessions"]:
            if useronly:
                retrieved_chunks += [x for x in session_entry if x["role"] == "user"]
            else:
                retrieved_chunks += [x for x in session_entry]

    elif retriever_type == "oracle-session":
        for session_id, session_entry in zip(entry["haystack_session_ids"], entry["haystack_sessions"]):
            if "answer" in session_id:
                if useronly:
                    retrieved_chunks.append([x for x in session_entry if x["role"] == "user"])
                else:
                    retrieved_chunks.append(session_entry)

    elif retriever_type == "oracle-turn":
        for session_id, session_entry in zip(entry["haystack_session_ids"], entry["haystack_sessions"]):
            if "answer" in session_id:
                if useronly:
                    retrieved_chunks += [x for x in session_entry if x["role"] == "user" and x.get("has_answer", False)]
                else:
                    retrieved_chunks += [x for x in session_entry if x.get("has_answer", False)]

    elif retriever_type == "flat-turn":
        for ret_result_entry in entry["retrieval_results"]["ranked_items"]:
            converted_corpus_id = "_".join(
                ret_result_entry["corpus_id"].replace("noans_", "answer_").split("_")[:-1]
            )
            converted_turn_id = int(
                ret_result_entry["corpus_id"].replace("noans_", "answer_").split("_")[-1]
            ) - 1
            try:
                cur_round_data = [corpusid2entry[converted_corpus_id][converted_turn_id]]
                converted_next_turn_id = converted_turn_id + 1
                if converted_next_turn_id < len(corpusid2entry[converted_corpus_id]):
                    cur_round_data.append(corpusid2entry[converted_corpus_id][converted_next_turn_id])
                retrieved_chunks.append(cur_round_data)
            except Exception:
                continue

    elif retriever_type == "rmm_memory":
        for mem_idx, ret_result_entry in enumerate(entry["retrieval_results"]["ranked_items"]):
            memory_text = build_rmm_memory_json(ret_result_entry, mem_idx)
            retrieved_chunks.append(memory_text)

    elif retriever_type == "flat-session":
        for ret_result_entry in entry["retrieval_results"]["ranked_items"]:
            session_key = ret_result_entry["corpus_id"].replace("noans_", "answer_")
            if useronly:
                retrieved_chunks.append([x for x in corpusid2entry[session_key] if x["role"] == "user"])
            else:
                retrieved_chunks.append(corpusid2entry[session_key])

    elif retriever_type == "no-retrieval":
        retrieved_chunks = []

    elif retriever_type == "gold":
        retrieved_chunks = []

    else:
        raise NotImplementedError(f"Unsupported retriever_type: {retriever_type}")

    if retriever_type in ["orig-turn", "orig-session", "rmm_memory"]:
        retrieved_chunks = retrieved_chunks[-topk_context:]
    else:
        retrieved_chunks = retrieved_chunks[:topk_context]

    retrieved_chunks_cleaned = []
    for retrieved_item in retrieved_chunks:
        try:
            session_entry = retrieved_item
            for turn_entry in session_entry:
                if isinstance(turn_entry, dict) and "has_answer" in turn_entry:
                    turn_entry.pop("has_answer")
            retrieved_chunks_cleaned.append(session_entry)
        except Exception:
            print("some error occured", flush=True)

    retrieved_chunks = retrieved_chunks_cleaned

    history_string = ""
    for i, chunk_entry in enumerate(retrieved_chunks):
        if history_format == "json":
            sess_string = "\n" + json.dumps(chunk_entry)
        elif history_format == "nl":
            sess_string = ""
            if isinstance(chunk_entry, list):
                for turn_entry in chunk_entry:
                    sess_string += f"\n\n{turn_entry['role']}: {turn_entry['content'].strip()}"
            else:
                sess_string += f"{chunk_entry['role']}: {chunk_entry['content'].strip()}"
        else:
            raise NotImplementedError

        if retriever_type in [
            "orig-session", "flat-session", "oracle-session",
            "orig-turn", "flat-turn", "oracle-turn", "rmm_memory"
        ]:
            history_string += f"\n### Session {i+1}:\nSession Content:\n{sess_string}\n"

    assert retriever_type in ["no-retrieval", "gold"] or history_string != ""

    if retriever_type == "no-retrieval":
        prompt = answer_prompt_template.format(question_string)
    elif retriever_type == "gold":
        gold_hint = format_gold_dimension(query_dimension, expected_category)
        prompt = answer_prompt_template.format(question_string, gold_hint)
    else:
        history_string = truncate_with_tiktoken(history_string, tokenizer, max_retrieval_length)
        prompt = answer_prompt_template.format(history_string, question_string)

    ## Location for drinking/explicit content queries 
    if entry.get("query_topic") == "Drinking/explicit content":
        prompt = "User Location: United States\n\n" + prompt

    return prompt


def count_completed_lines(path: str) -> int:
    if not os.path.exists(path):
        return 0
    with open(path, "r") as f:
        return sum(1 for line in f if line.strip())


def main(args):
    try:
        in_data = json.load(open(args.in_file))
    except Exception:
        in_data = [json.loads(line) for line in open(args.in_file).readlines()]

    # in_data = in_data[:1]

    # in_data = [entry for entry in in_data if entry.get("query_topic") == "Drinking/explicit content"] 
    # print(len(in_data))

    os.makedirs(args.out_dir, exist_ok=True)

    out_file = os.path.join(
        args.out_dir,
        "{}_{}_testlog_top{}context_{}format_useronly{}{}.jsonl".format(
            args.model_name.replace("/", "_"),
            args.retriever_type,
            args.topk_context,
            args.history_format,
            args.useronly,
            args.out_file_suffix,
        ),
    )

    completed = count_completed_lines(out_file)
    print(f"Resume mode: found {completed} completed entries in {out_file}", flush=True)

    model2maxlength = {
        "gpt-4o": 128000,
        "gpt-4o-2024-08-06": 128000,
        "gpt-4o-mini": 128000,
        "gpt-4o-mini-2024-07-18": 128000,
        "gpt-4.1": 128000,
        "gpt-4.1-mini": 128000,
        "qwen": 128000,
    }

    model_name_lower = args.model_name.lower()
    model_max_length = 128000
    for k, v in model2maxlength.items():
        if k in model_name_lower:
            model_max_length = v
            break

    tokenizer = tiktoken.get_encoding("o200k_base")

    total_prompt_tokens = 0
    total_completion_tokens = 0

    with open(out_file, "a") as out_f:
        for idx, entry in enumerate(
            tqdm(in_data, desc="running generations", total=len(in_data))
        ):
            if idx < completed:
                continue

            gen_length = args.gen_length if args.gen_length is not None else 800
            max_retrieval_length = model_max_length - gen_length - 1000

            prompt = prepare_prompt(
                entry=entry,
                retriever_type=args.retriever_type,
                topk_context=args.topk_context,
                useronly=(args.useronly == "true"),
                history_format=args.history_format,
                tokenizer=tokenizer,
                max_retrieval_length=max_retrieval_length,
            )

            # print(prompt)

            try:
                answer, prompt_tks, completion_tks = query_llm(
                    model_path=args.model_name,
                    user_prompt=prompt,
                    temperature=0,
                    max_tokens=gen_length,
                    print_tokens=True,
                )

                # print(answer)

                total_prompt_tokens += prompt_tks or 0
                total_completion_tokens += completion_tks or 0

                output_entry = copy.deepcopy(entry)
                output_entry["hypothesis"] = answer
                output_entry.pop("haystack_sessions", None)
                output_entry.pop("haystack_session_ids", None)

                print(json.dumps(output_entry), file=out_f, flush=True)

            except Exception as e:
                print(f"Stopped at idx={idx} with error: {repr(e)}", flush=True)
                break

            # break

    print(f"Total prompt tokens (this run): {total_prompt_tokens}", flush=True)
    print(f"Total completion tokens (this run): {total_completion_tokens}", flush=True)
    print(f"Total tokens (this run): {total_prompt_tokens + total_completion_tokens}", flush=True)


if __name__ == "__main__":
    args = parse_args()
    check_args(args)
    main(args)