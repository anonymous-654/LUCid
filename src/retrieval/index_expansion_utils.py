def fetch_expansion_from_cache(index_expansion_result_cache, cur_sess_id):
    processed_id = cur_sess_id.replace("answer_", "").replace("noans_", "")
    try:
        cur_expansion = index_expansion_result_cache[processed_id]
    except Exception:
        cur_expansion = None

    if isinstance(cur_expansion, str):
        cur_expansion = [cur_expansion]

    return cur_expansion


def resolve_session_userfact_expansion(
    resolution_strategy,
    existing_corpus,
    existing_corpus_ids,
    cur_item_expansions,
    cur_sess_id,
):
    if cur_item_expansions is None:
        cur_item_expansions = [""]

    cur_item_expansions = [str(x) for x in cur_item_expansions]

    if "split" not in resolution_strategy:
        if cur_item_expansions:
            cur_item_expansions = [" ".join(cur_item_expansions)]
        else:
            cur_item_expansions = []

    if "separate" in resolution_strategy:
        existing_corpus += [str(x) for x in cur_item_expansions]
        existing_corpus_ids += [cur_sess_id for _ in cur_item_expansions]

    elif "merge" in resolution_strategy or "replace" in resolution_strategy:
        out_corpus, out_corpus_ids = [], []

        for i in range(len(existing_corpus_ids)):
            if existing_corpus_ids[i] == cur_sess_id:
                for expansion_item in cur_item_expansions:
                    if "merge" in resolution_strategy:
                        merged = f"{expansion_item} {existing_corpus[i]}".strip()
                        out_corpus.append(merged)
                    elif "replace" in resolution_strategy:
                        out_corpus.append(expansion_item)
                    else:
                        raise NotImplementedError

                    out_corpus_ids.append(existing_corpus_ids[i])
            else:
                out_corpus.append(existing_corpus[i])
                out_corpus_ids.append(existing_corpus_ids[i])

        existing_corpus, existing_corpus_ids = out_corpus, out_corpus_ids

    else:
        raise NotImplementedError(
            f"Unsupported resolution strategy: {resolution_strategy}"
        )

    return existing_corpus, existing_corpus_ids