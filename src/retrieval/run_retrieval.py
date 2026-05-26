import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from rank_bm25 import BM25Okapi
from tqdm import tqdm
from sklearn.preprocessing import normalize
from transformers import AutoModel, AutoTokenizer

from src.retrieval.eval_utils import evaluate_retrieval, evaluate_retrieval_turn2session
from src.retrieval.index_expansion_utils import (
    fetch_expansion_from_cache,
    resolve_session_userfact_expansion,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_file", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--outfile_prefix", type=str, default=None)

    parser.add_argument(
        "--retriever",
        type=str,
        required=True,
        choices=['flat-bm25', 'flat-contriever', 'flat-stella', 'flat-gte', 'oracle'],
    )
    parser.add_argument(
        "--granularity",
        type=str,
        required=True,
        choices=["session", "turn"],
    )

    # session-userfact expansion only
    parser.add_argument(
        "--index_expansion_result_cache",
        type=str,
        default=None,
        help="Path to precomputed session-userfact cache JSON",
    )
    parser.add_argument(
        "--index_expansion_result_join_mode",
        type=str,
        default="none",
        choices=[
            "separate",
            "split-separate",
            "merge",
            "split-merge",
            "replace",
            "split-replace",
            "none",
        ],
    )

    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Path to cache dir",
    )

    return parser.parse_args()


def check_args(args):
    print(args)

    use_expansion = (
        args.index_expansion_result_cache is not None
        and str(args.index_expansion_result_cache).lower() != "none"
    )

    if use_expansion:
        assert args.index_expansion_result_join_mode != "none", (
            "When using session-userfact expansion, "
            "--index_expansion_result_join_mode must not be 'none'."
        )
        assert os.path.isfile(args.index_expansion_result_cache), (
            f"Expansion cache not found: {args.index_expansion_result_cache}"
        )


def get_outfile_prefix(args):
    if args.outfile_prefix is not None and args.outfile_prefix.lower() != "none":
        return args.outfile_prefix
    return os.path.basename(args.in_file)


class DenseRetrievalMaster:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.retriever_model = None
        self.prepare_retriever()

    def prepare_retriever(self):
        self.retriever_model = None
        
        if self.args.retriever == 'flat-contriever':
            model = AutoModel.from_pretrained('facebook/contriever').to(self.device)
            tokenizer = AutoTokenizer.from_pretrained('facebook/contriever')
            self.retriever_model = (tokenizer, model)

        elif self.args.retriever == 'flat-stella':
            # model_dir = self.args.cache_dir + "/dunzhang_stella_en_1.5B_v5"
            model_dir = self.args.cache_dir
            vector_dim = 1024
            vector_linear_directory = f"2_Dense_{vector_dim}"
            model = AutoModel.from_pretrained(model_dir, trust_remote_code=True).to(self.device)
            model.eval()
            tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
            vector_linear = torch.nn.Linear(in_features=model.config.hidden_size, out_features=vector_dim).to(self.device)
            vector_linear_dict = {
                k.replace("linear.", ""): v for k, v in
                torch.load(os.path.join(model_dir, f"{vector_linear_directory}/pytorch_model.bin")).items()
            }
            vector_linear.load_state_dict(vector_linear_dict)
            vector_linear.to(self.device)
            self.retriever_model = (tokenizer, model, vector_linear)
            
        elif self.args.retriever == 'flat-gte':
            tokenizer = AutoTokenizer.from_pretrained('Alibaba-NLP/gte-Qwen2-7B-instruct', trust_remote_code=True)
            model = AutoModel.from_pretrained('Alibaba-NLP/gte-Qwen2-7B-instruct', trust_remote_code=True).to(self.device)
            model.eval()
            self.retriever_model = (tokenizer, model)

    @staticmethod
    def mean_pooling(token_embeddings, mask):
        token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.0)
        sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
        return sentence_embeddings

    def run_flat_retrieval(self, query, retriever, corpus):
        if retriever == 'flat-bm25':
            tokenized_corpus = [doc.split(" ") for doc in corpus]
            # tokenized_torpus = word_tokenize(corpus)
            bm25 = BM25Okapi(tokenized_corpus)
            scores = bm25.get_scores(query.split(" "))
            return np.argsort(scores)[::-1]

        elif retriever in ['flat-contriever', 'flat-stella', 'flat-gte']:
            model2bsz = {'flat-contriever': 128, 'flat-stella': 64, 'flat-gte': 1}
            bsz = model2bsz[retriever]
            
            if retriever == 'flat-contriever':
                tokenizer, model = self.retriever_model
                def mean_pooling(token_embeddings, mask):
                    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
                    sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
                    return sentence_embeddings

                with torch.no_grad():
                    inputs = tokenizer([query], padding=True, truncation=True, return_tensors='pt')
                    inputs = {k: v.to(model.device) for k, v in inputs.items()}
                    outputs = model(**inputs)
                    query_vectors = mean_pooling(outputs[0], inputs['attention_mask']).detach().cpu()
                    all_docs_vectors = []
                    dataloader = DataLoader(corpus, batch_size=bsz, shuffle=False)
                    for batch in dataloader:
                        inputs = tokenizer(batch, padding=True, truncation=True, return_tensors='pt')
                        inputs = {k: v.to(model.device) for k, v in inputs.items()}
                        outputs = model(**inputs)
                        cur_docs_vectors = mean_pooling(outputs[0], inputs['attention_mask']).detach().cpu()
                        all_docs_vectors.append(cur_docs_vectors)
                    all_docs_vectors = np.concatenate(all_docs_vectors, axis=0)
                    scores = (query_vectors @ all_docs_vectors.T).squeeze()
                
            elif retriever == 'flat-stella':
                tokenizer, model, vector_linear = self.retriever_model
                with torch.no_grad():
                    input_data = tokenizer([query], padding="longest", truncation=True, max_length=512, return_tensors="pt")
                    input_data = {k: v.to(model.device) for k, v in input_data.items()}
                    attention_mask = input_data["attention_mask"]
                    last_hidden_state = model(**input_data)[0]
                    last_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
                    query_vectors = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
                    query_vectors = normalize(vector_linear(query_vectors).detach().cpu())
                with torch.no_grad():
                    all_docs_vectors = []
                    dataloader = DataLoader(corpus, batch_size=bsz, shuffle=False)
                    for batch in dataloader:
                        input_data = tokenizer(batch, padding="longest", truncation=True, max_length=512, return_tensors="pt")
                        input_data = {k: v.to(model.device) for k, v in input_data.items()}
                        attention_mask = input_data["attention_mask"]
                        last_hidden_state = model(**input_data)[0]
                        last_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
                        docs_vectors = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
                        docs_vectors = normalize(vector_linear(docs_vectors).detach().cpu())
                        all_docs_vectors.append(docs_vectors)
                    all_docs_vectors = np.concatenate(all_docs_vectors, axis=0)
                scores = torch.tensor((query_vectors @ all_docs_vectors.T).squeeze())

            elif retriever == 'flat-gte':
                def last_token_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
                    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
                    if left_padding:
                        return last_hidden_states[:, -1]
                    else:
                        sequence_lengths = attention_mask.sum(dim=1) - 1
                        batch_size = last_hidden_states.shape[0]
                        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]
                    
                def get_detailed_instruct(task_description: str, query: str) -> str:
                    return f'Instruction: {task_description}\nQuery: {query}'

                tokenizer, model = self.retriever_model
                task = 'Given a query about personal information, retrieve relevant chat history that answer the query.'
                with torch.no_grad():
                    all_vectors = []
                    dataloader = DataLoader([get_detailed_instruct(task, query)] + corpus, batch_size=bsz, shuffle=False)
                    for batch in dataloader:
                        batch_dict = tokenizer(batch, max_length=8192, padding=True, truncation=True, return_tensors='pt')
                        batch_dict = {k: v.to(model.device) for k, v in batch_dict.items()}
                        outputs = model(**batch_dict)
                        embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
                        all_vectors.append(embeddings)
                all_vectors = torch.cat(all_vectors, dim=0)
                all_vectors = F.normalize(all_vectors, p=2, dim=1)
                scores = (all_vectors[:1] @ all_vectors[1:].T).squeeze()

            else:
                raise NotImplementedError

            return scores.argsort(descending=True)
        
        else:
            raise NotImplementedError


def process_item_flat_index(data, granularity, sess_id):
    corpus = []

    if granularity == "session":
        text = " ".join([turn["content"] for turn in data if turn["role"] == "user"])
        corpus.append(text)
        ids = [sess_id]

        if "answer" in sess_id and all(
            [not turn["has_answer"] for turn in data if turn["role"] == "user"]
        ):
            ids = [sess_id.replace("answer", "noans")]

    elif granularity == "turn":
        ids = []
        for i_turn, turn in enumerate(data):
            if turn["role"] == "user":
                corpus.append(turn["content"])
                turn_id = f"{sess_id}_{i_turn + 1}"

                if "answer" not in sess_id:
                    ids.append(turn_id)
                else:
                    assert "has_answer" in turn
                    assert turn["has_answer"] in [True, False]
                    if turn["has_answer"]:
                        ids.append(turn_id)
                    else:
                        ids.append(turn_id.replace("answer", "noans"))
                        assert "answer" not in ids[-1]
    else:
        raise NotImplementedError

    return corpus, ids


def add_session_userfact_expansion(entry, args, corpus, corpus_ids, expansion_cache):
    if expansion_cache is None:
        return corpus, corpus_ids

    for cur_sess_id, _sess_entry in zip(
        entry["haystack_session_ids"], entry["haystack_sessions"]
    ):
        cur_item_expansions = fetch_expansion_from_cache(expansion_cache, cur_sess_id)
        corpus, corpus_ids = resolve_session_userfact_expansion(
            resolution_strategy=args.index_expansion_result_join_mode,
            existing_corpus=corpus,
            existing_corpus_ids=corpus_ids,
            cur_item_expansions=cur_item_expansions,
            cur_sess_id=cur_sess_id,
        )

    return corpus, corpus_ids


def batch_get_retrieved_context_and_eval(entry_list, args, retriever_master, expansion_cache=None):
    # if args.retriever in ['flat-bm25', 'flat-contriever', 'flat-stella', 'flat-gte', 'oracle']:
    #     retriever_master = DenseRetrievalMaster(args)
    # else:
    #     raise NotImplementedError

    results = []

    for entry in tqdm(entry_list):
        corpus, corpus_ids = [], []

        for cur_sess_id, sess_entry in zip(
            entry["haystack_session_ids"], entry["haystack_sessions"]
        ):
            cur_items, cur_ids = process_item_flat_index(
                sess_entry,
                args.granularity,
                cur_sess_id,
            )
            corpus += cur_items
            corpus_ids += cur_ids

        use_expansion = (
            expansion_cache is not None
            and args.index_expansion_result_join_mode != "none"
        )
        if use_expansion:
            corpus, corpus_ids = add_session_userfact_expansion(
                entry, args, corpus, corpus_ids, expansion_cache
            )

        correct_docs = list(set([doc_id for doc_id in corpus_ids if "answer" in doc_id]))
        query = entry["query"]

        if args.retriever in ['flat-bm25', 'flat-contriever', 'flat-stella', 'flat-gte']:
            rankings = retriever_master.run_flat_retrieval(query, args.retriever, corpus)
        elif args.retriever == "oracle":
            correct_idx, incorrect_idx = [], []
            for i_doc, cid in enumerate(corpus_ids):
                if cid in correct_docs:
                    correct_idx.append(i_doc)
                else:
                    incorrect_idx.append(i_doc)
            rankings = correct_idx + incorrect_idx
        else:
            raise NotImplementedError(f"Retriever not supported: {args.retriever}")

        cur_results = dict(entry)

        cur_results["retrieval_results"] = {
            "query": query,
            "ranked_items": [
                {"corpus_id": corpus_ids[rid], "text": corpus[rid]}
                for rid in rankings
            ],
            "metrics": {"session": {}, "turn": {}},
        }

        KS = [1, 3, 5, 10, 30, 50]
        for k in KS:
            recall_any, recall_all, ndcg_any, recall_ge_n = evaluate_retrieval(
                rankings, correct_docs, corpus_ids, k=k
            )
            cur_results["retrieval_results"]["metrics"][args.granularity].update({
                f"recall_any@{k}": recall_any,
                f"recall_all@{k}": recall_all,
                f"recall_ge_n@{k}": recall_ge_n,
                f"ndcg_any@{k}": ndcg_any,
            })

            if args.granularity == "turn":
                recall_any, recall_all, ndcg_any, recall_ge_n = evaluate_retrieval_turn2session(
                    rankings, correct_docs, corpus_ids, k=k
                )
                cur_results["retrieval_results"]["metrics"]["session"].update({
                    f"recall_any@{k}": recall_any,
                    f"recall_all@{k}": recall_all,
                    f"recall_ge_n@{k}": recall_ge_n,
                    f"ndcg_any@{k}": ndcg_any,
                })

        cur_results.pop("haystack_sessions", None)
        cur_results.pop("haystack_session_ids", None)

        results.append(cur_results)

    return results


def main(args):
    check_args(args)
    os.makedirs(args.out_dir, exist_ok=True)

    in_data = json.load(open(args.in_file))
    # in_data = in_data[:2]
    size = len(in_data)

    expansion_cache = None
    if (
        args.index_expansion_result_cache is not None
        and str(args.index_expansion_result_cache).lower() != "none"
    ):
        expansion_cache = json.load(open(args.index_expansion_result_cache))
        print(f"Loaded session-userfact expansions from {args.index_expansion_result_cache}")

    outfile_prefix = get_outfile_prefix(args)
    out_file = (
        f"{args.out_dir}/{outfile_prefix}_retrievallog_"
        f"{args.granularity}_{args.retriever}_{size}"
    )
    print(out_file)

    # retriever_master = DenseRetrievalMaster(args)

    if args.retriever in ['flat-bm25', 'flat-contriever', 'flat-stella', 'flat-gte', 'oracle']:
        retriever_master = DenseRetrievalMaster(args)
    else:
        raise NotImplementedError

    results = []
    for entry in tqdm(in_data, desc="Processing examples", total=len(in_data)):
        batch_result = batch_get_retrieved_context_and_eval(
            [entry],
            args,
            retriever_master,
            expansion_cache=expansion_cache,
        )
        results.extend(batch_result)

    averaged_results = {"session": {}, "turn": {}}
    ignored_qs_no_target = set()

    for t in ["session", "turn"]:
        if not results or not results[0]["retrieval_results"]["metrics"][t]:
            continue

        for metric_name in results[0]["retrieval_results"]["metrics"][t]:
            try:
                results_list = []
                for eval_entry in results:
                    if not any(
                        ("has_answer" in turn and turn["has_answer"])
                        for turn in [
                            x
                            for y in eval_entry["haystack_sessions"]
                            for x in y
                            if x["role"] == "user"
                        ]
                    ):
                        ignored_qs_no_target.add(
                            eval_entry.get("question", eval_entry.get("query", "unknown"))
                        )
                        continue

                    results_list.append(
                        eval_entry["retrieval_results"]["metrics"][t][metric_name]
                    )

                averaged_results[t][metric_name] = np.mean(results_list)
            except Exception:
                continue

    print(
        f"Additionally ignored {len(ignored_qs_no_target)} instances due to no "
        f"target turns from the user side: {ignored_qs_no_target}"
    )
    print(json.dumps(averaged_results))

    with open(out_file, "w") as out_f:
        for entry in results:
            print(json.dumps(entry), file=out_f)


if __name__ == "__main__":
    args = parse_args()
    main(args)