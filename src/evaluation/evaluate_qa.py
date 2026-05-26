import argparse
import json
import os
import string
from typing import List

from tqdm import tqdm

from src.utils.query import query_gpt, query_llm


def get_letter_value(letter:str, references: List):
    references = references.copy()
    references.append("None of the above")
    letter = letter.strip().upper()
    options = list(string.ascii_uppercase)[:len(references)]
    
    if letter in options:
        index = options.index(letter)
        return references[index]
    else:
        return None


def extract_references(ref_list):
    ref_list = ref_list.copy()  # avoid modifying the original list
    ref_list.append("None of the above")
    options = list(string.ascii_uppercase)[:len(ref_list)]
    ref_text = ""
    for i, option in enumerate(ref_list):
        ref_text += f"\n{options[i]}. {ref_list[i]}"
    return ref_text, options, ref_list

def prepare_evaluation_prompt(question: str, answer: str, references, label: str):
    ref_text, options, full_ref_list = extract_references(references)


    # Get the letter corresponding to the label
    try:
        label_index = full_ref_list.index(label)
        label_letter = options[label_index]
    except ValueError:
        label_letter = None  # or raise error if preferred

    prompt = (
        f"Question: {question}\n"
        f"Answer: {answer}\n\n"
        "Which of the following domains does the answer best apply to?\n"
        f"{ref_text}\n\n"
        "Choose the best option based on the content of the answer alone. "
        "Return only the option letter (e.g., 'A')."
    )
    # print(prompt)
    return prompt, label_letter

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_file', type=str, required=True)
    parser.add_argument('--out_dir', type=str, required=True)
        
    parser.add_argument('--model_name', type=str, required=True)
    
    return parser.parse_args()


def main(args):
    # setup
    try:
        in_data = json.load(open(args.in_file))
    except:
        in_data = [json.loads(line) for line in open(args.in_file).readlines()]

    # Extract desired components from the input path
    oracle_dir = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(args.in_file))))  # "oracle-session"
    suffix = os.path.basename(args.in_file).replace('_testlog_', '')               # "top5context_jsonformat_useronlyfalse"

    # Construct output filename
    out_file = os.path.join(
        args.out_dir,
        f"eval-results-{args.model_name}-{oracle_dir}-{suffix}"
    )
    print(out_file)
    # out_file = args.out_dir + '/' + 'eval-results-{}'.format(args.model_name)
    out_f = open(out_file, 'w')

    results = []
    print(len(in_data))
    # in_data = in_data[:4]
    for entry in tqdm(in_data, desc='running generations', total = len(in_data)):
        try:
            prompt, label_letter= prepare_evaluation_prompt(question = entry['question'], answer = entry['hypothesis'],
            references=entry['domains'], label=entry['expected_domain'])
            # answer, _, _ = query_llm(user_prompt=prompt)
            answer, _, _ = query_gpt(user_prompt=prompt)
            answer_domain = get_letter_value(answer, entry['domains'])
            # print(label_letter, answer)
            is_correct = 0
            if label_letter == answer:
                is_correct = 1

            results.append(is_correct)
            print(json.dumps({"main_word": entry['main_word'], 'question': entry['question'], 
            'expected_domain': entry['expected_domain'], 'domains': entry['domains'], 'hypothesis':entry['hypothesis'],  'answer_domain':answer_domain, 'is_correct': is_correct}), file=out_f, flush=True)
        except Exception as e:
            print('One exception captured', repr(e))
            continue

    out_f.close()

    accuracy = sum(results) / len(results)
    print(f"Accuracy: {accuracy:.4f}")


if __name__ == '__main__':
    args = parse_args()
    main(args)
