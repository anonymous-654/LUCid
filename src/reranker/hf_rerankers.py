import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
)


class BGESequenceReranker:
    """
    For:
      - BAAI/bge-reranker-v2-m3
    """

    def __init__(self, model_name: str, device: str = None, max_length: int = 512):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )

        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=dtype,
        ).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def score_pairs(self, query: str, docs, batch_size: int = 8):
        scores = []

        for i in range(0, len(docs), batch_size):
            batch_docs = docs[i : i + batch_size]
            pairs = [[query, doc] for doc in batch_docs]

            inputs = self.tokenizer(
                pairs,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            outputs = self.model(**inputs, return_dict=True)
            logits = outputs.logits

            if logits.dim() == 2 and logits.size(-1) == 1:
                batch_scores = logits.squeeze(-1)
            elif logits.dim() == 1:
                batch_scores = logits
            else:
                batch_scores = logits[:, 0]

            scores.extend(batch_scores.detach().float().cpu().tolist())

        return scores


class BGEGemmaReranker:
    """
    For:
      - BAAI/bge-reranker-v2-gemma

    Uses a CausalLM scoring setup:
    score = logits[:, -1, yes_token_id]
    """

    DEFAULT_PROMPT = (
    "Given a user query A and a past session B, determine whether the session "
    "contains information useful for generating a personalized, safe and appropriate response to the query. "
    "Answer 'yes' or 'no'.")

    # DEFAULT_PROMPT = ( "Given a query A and a passage B, determine whether the passage " "contains an answer to the query by providing a prediction of either " "'Yes' or 'No'." )

    def __init__(self, model_name: str, device: str = None, max_length: int = 1024):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )

        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=dtype,
        ).to(self.device)
        self.model.eval()

        yes_ids = self.tokenizer("Yes", add_special_tokens=False)["input_ids"]
        if len(yes_ids) != 1:
            raise ValueError(
                f"'Yes' must tokenize to a single token for {model_name}, got {yes_ids}"
            )
        self.yes_token_id = yes_ids[0]

    def _build_inputs(self, pairs, prompt=None):
        prompt = prompt or self.DEFAULT_PROMPT
        tokenizer = self.tokenizer
        max_length = self.max_length
        sep = "\n"

        prompt_ids = tokenizer(
            prompt,
            add_special_tokens=False,
            return_tensors=None,
        )["input_ids"]

        sep_ids = tokenizer(
            sep,
            add_special_tokens=False,
            return_tensors=None,
        )["input_ids"]

        items = []
        for query, passage in pairs:
            query_ids = tokenizer(
                f"A: {query}",
                add_special_tokens=False,
                truncation=True,
                max_length=max_length * 3 // 4,
                return_tensors=None,
            )["input_ids"]

            passage_ids = tokenizer(
                f"B: {passage}",
                add_special_tokens=False,
                truncation=True,
                max_length=max_length,
                return_tensors=None,
            )["input_ids"]

            item = tokenizer.prepare_for_model(
                [tokenizer.bos_token_id] + query_ids,
                sep_ids + passage_ids,
                truncation="only_second",
                max_length=max_length,
                padding=False,
                return_attention_mask=False,
                return_token_type_ids=False,
                add_special_tokens=False,
            )

            item_input_ids = item["input_ids"] + sep_ids + prompt_ids
            items.append(
                {
                    "input_ids": item_input_ids,
                    "attention_mask": [1] * len(item_input_ids),
                }
            )

        padded = tokenizer.pad(
            items,
            padding=True,
            pad_to_multiple_of=8,
            return_tensors="pt",
        )
        return padded

    @torch.no_grad()
    def score_pairs(self, query: str, docs, batch_size: int = 4):
        scores = []

        for i in range(0, len(docs), batch_size):
            batch_docs = docs[i : i + batch_size]
            pairs = [[query, doc] for doc in batch_docs]

            inputs = self._build_inputs(pairs)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            outputs = self.model(**inputs, return_dict=True)
            logits = outputs.logits

            batch_scores = logits[:, -1, self.yes_token_id].float().cpu().tolist()
            scores.extend(batch_scores)

        return scores
    

class Qwen3CausalReranker:
    """
    For:
      - Qwen/Qwen3-Reranker-0.6B
      - Qwen/Qwen3-Reranker-4B
      - Qwen/Qwen3-Reranker-8B

    Uses the Qwen3 reranker scoring pattern:
      score = P("yes") / (P("yes") + P("no"))
    from the final logits.
    """

    SYSTEM_TEXT = (
        'Judge whether the Document meets the requirements based on the Query '
        'and the Instruct provided. Note that the answer can only be "yes" or "no".'
    )

    DEFAULT_INSTRUCTION = (
        "Given a user query, retrieve past sessions that are most useful for generating a personalized, safe and appropriate response for this user."
    )
    # "Given a web search query, retrieve relevant passages that answer the query"

    def __init__(
        self,
        model_name: str,
        device: str = None,
        max_length: int = 8192,
        instruction: str = None,
    ):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.instruction = instruction or self.DEFAULT_INSTRUCTION

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side="left",
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=dtype,
        ).to(self.device)
        self.model.eval()

        yes_ids = self.tokenizer("yes", add_special_tokens=False)["input_ids"]
        no_ids = self.tokenizer("no", add_special_tokens=False)["input_ids"]

        if len(yes_ids) != 1 or len(no_ids) != 1:
            raise ValueError(
                f"'yes'/'no' must each tokenize to a single token for {model_name}, "
                f"got yes={yes_ids}, no={no_ids}"
            )

        self.yes_token_id = yes_ids[0]
        self.no_token_id = no_ids[0]

        # Per Qwen model card format
        self.suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self.suffix_tokens = self.tokenizer.encode(self.suffix, add_special_tokens=False)

    def _format_pair(self, query: str, doc: str):
        return [
            {"role": "system", "content": self.SYSTEM_TEXT},
            {
                "role": "user",
                "content": (
                    f"<Instruct>: {self.instruction}\n"
                    f"<Query>: {query}\n"
                    f"<Document>: {doc}"
                ),
            },
        ]

    def _build_inputs(self, pairs):
        tokenized = []
        max_body_len = self.max_length - len(self.suffix_tokens)

        for query, doc in pairs:
            messages = self._format_pair(query, doc)
            ids = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
                enable_thinking=False,
            )
            ids = ids[:max_body_len] + self.suffix_tokens
            tokenized.append({"input_ids": ids, "attention_mask": [1] * len(ids)})

        padded = self.tokenizer.pad(
            tokenized,
            padding=True,
            pad_to_multiple_of=8,
            return_tensors="pt",
        )
        return padded

    @torch.no_grad()
    def score_pairs(self, query: str, docs, batch_size: int = 4):
        scores = []

        for i in range(0, len(docs), batch_size):
            batch_docs = docs[i : i + batch_size]
            pairs = [(query, doc) for doc in batch_docs]

            inputs = self._build_inputs(pairs)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            outputs = self.model(**inputs, return_dict=True)
            final_logits = outputs.logits[:, -1, :]

            yes_logits = final_logits[:, self.yes_token_id]
            no_logits = final_logits[:, self.no_token_id]

            yn = torch.stack([no_logits, yes_logits], dim=1)
            yn = torch.nn.functional.log_softmax(yn, dim=1)
            batch_scores = yn[:, 1].exp().float().cpu().tolist()

            scores.extend(batch_scores)

        return scores