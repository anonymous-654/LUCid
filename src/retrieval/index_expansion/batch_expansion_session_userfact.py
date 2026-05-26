import os
import json
from pathlib import Path
from tqdm import tqdm
from openai import OpenAI
import openai
import backoff


@backoff.on_exception(backoff.constant, (openai.RateLimitError,), interval=5)
def chat_completions_with_backoff(client, **kwargs):
    return client.chat.completions.create(**kwargs)

NODE_HOSTNAME = "gl1529.arc-ts.umich.edu" 

client = OpenAI(
    base_url=f"http://{NODE_HOSTNAME}:8001/v1",
    api_key="EMPTY"
)


def extract_session_userfact(sess_entry, model_name, examples=None):
    system_prompt = (
        "You will be given a list of messages from a human user to an AI assistant. "
        "Extract all the personal information, life events, experience, and preferences "
        "related to the user. Make sure you include all details such as life events, "
        "personal experience, preferences, specific numbers, locations, or dates. "
        "State each piece of information in a simple sentence. Put these sentences in "
        "a json list, each element being a standalone personal fact about the user. "
        "Minimize the coreference across the facts, e.g., replace pronouns with actual "
        "entities. If there is no specific events, personal information, or preference "
        "mentioned, just generate an empty list."
    )

    user_prompt = (
        "Human user messages:\n{}\n\n"
        "Personal facts about the user (a list of strings in json format; "
        "do not generate anything else):"
    )

    dialogue_string = ""
    for turn_entry in sess_entry:
        if turn_entry["role"] == "user":
            dialogue_string += f"\n{turn_entry['role']}：{turn_entry['content']}"

    summarization_prompt = user_prompt.format(dialogue_string)

    if examples is None:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": summarization_prompt},
        ]
    else:
        messages = [{"role": "system", "content": system_prompt}]
        for example_input_dialogue_string, example_output in examples:
            messages += [
                {"role": "user", "content": user_prompt.format(example_input_dialogue_string)},
                {"role": "assistant", "content": example_output},
            ]
        messages += [{"role": "user", "content": summarization_prompt}]

    kwargs = {
        "model": model_name,
        "messages": messages,
        "max_tokens":512,
        "temperature":0.7,
        "extra_body": {
            "top_k": 20,
            "chat_template_kwargs": {"enable_thinking": False},}
    }

    completion = chat_completions_with_backoff(client, **kwargs)
    print(completion)

    try:
        out_string = completion.choices[0].message.content.strip()
        out_string = out_string.replace("```json", "")
        out_string = out_string.replace("```", "")
        return json.loads(out_string.strip())
    except Exception:
        print("Failed to parse model output:")
        print(completion.choices[0].message.content)
        return None


if __name__ == "__main__":
    model_name = "Qwen/Qwen3.5-27B-FP8"
    mode = "ICL"   # zero-shot, ICL
    assert mode in ["zero-shot", "ICL"]

    # current_dir = os.getcwd()
    in_file = "data/lucid_b.json"
    # out_dir = os.path.join(current_dir, "index_expansion_logs")
    current_dir = Path(__file__).resolve().parent
    out_dir = current_dir / "index_expansion_logs"
    os.makedirs(out_dir, exist_ok=True)

    cache_file = os.path.join(
        out_dir,
        os.path.basename(in_file) + f".session-userfact.{mode}.json"
    )

    if os.path.isfile(cache_file):
        with open(cache_file, "r") as f:
            data = json.load(f)
        print("Loaded:", cache_file)
    else:
        data = {}

    with open(in_file, "r") as f:
        in_data = json.load(f)

    # in_data = in_data[:1]
    # Build unique sessions from your current retrieval-format data
    todo_sessions = {}
    for entry in tqdm(in_data, desc="Processing examples", total=len(in_data)):
        for sess_id, sess in zip(entry["haystack_session_ids"], entry["haystack_sessions"]):
            base_sess_id = sess_id.replace("answer_", "").replace("noans_", "")
            if base_sess_id not in todo_sessions:
                todo_sessions[base_sess_id] = sess

    todo_sessions = [(sid, sess) for sid, sess in todo_sessions.items() if sid not in data]

    # todo_sessions = todo_sessions[:2]

    n_done, save_interval = 0, 500
    for sess_id, sess in tqdm(todo_sessions):
        if mode == "zero-shot":
            expansion = extract_session_userfact(sess, model_name, examples=None)
        else:
            examples = [
                (
                    "\nuser：What impact have recent economic developments had on Oxford's unique blend of old-world charm and modern innovation?\nuser：How has the city goverment responded to the economic challenges faced by the hospitality and tourism industries?\nuser：What specific measures has the city government taken to ensure the safety of tourists and locals amidst the pandemic?",
                    json.dumps([]),
                ),
                (
                    "\nuser：Could you explain the process of optimizing a website for search engines?\nuser：Do you have any tips for creating high-quality content that can attract more traffic?\nuser：These tips are very helpful. Is there a specific length or format that works best for creating content?\nuser：That makes sense! I'm also curious, how important is it to update old content on my website? Is it worth the effort?",
                    json.dumps(["The user is interested in optimizing their website."]),
                ),
                (
                    "\nuser：How did the British Empire expand and decline over the centuries?\nuser：It's interesting how the legacy of the British Empire still affects many countries today. Do you think it was ultimately a positive or negative force in the world?\nuser：It's interesting to learn about the impact of the British Empire, makes me wonder how different the world would be without it.\nuser：It's fascinating how the British Empire had such a far-reaching impact on the world. I wonder if there are any other empires in history that have left such a mark?",
                    json.dumps([]),
                ),
                (
                    "What techniques or tools can writers use to create a compelling and authentic character arc for their protagonists? I especially like the idea of using supporting characters to facilitate the protagonist's growth. Do you have any tips for making sure those supporting characters feel authentic and well-rounded? I think having well-rounded supporting characters can really elevate a story. Do you have any favorite examples of stories that have done this well?",
                    json.dumps([
                        "The user likes the idea of using supporting characters to facilitate the protagonist's growth.",
                        "The user thinks having well-rounded supporting characters can really elevate a story."
                    ]),
                ),
                (
                    "\nuser：What are the most iconic monuments to see in Washington, D.C.?\nuser：Can you recommend a good burger joint in D.C. near these monuments?\nuser：Hmm, all of those burger joints sound pretty generic. Do you have any recommendations for a more unique burger spot near the monuments in D.C.?\nuser：I don't know, Lucky Buns and Duke's Grocery sounds a little too fancy for a burger joint. I just want a good old-fashioned burger.\nuser：None of those classic burger joints sound good to me. How about something completely out of the box and unique?",
                    json.dumps([]),
                ),
            ]
            expansion = extract_session_userfact(sess, model_name, examples=examples)

        data[sess_id] = expansion
        print({sess_id: expansion})

        n_done += 1
        if n_done % save_interval == 0:
            with open(cache_file, "w") as f:
                json.dump(data, f, indent=2)

    with open(cache_file, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Saved cache to: {cache_file}")
