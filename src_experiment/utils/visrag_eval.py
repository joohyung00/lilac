

def eval_mrr(qrel, run, cutoff = 10):
    
    """
    Compute MRR@cutoff manually.
    """
    mrr = 0.0
    num_ranked_q = 0
    results = {}
    
    for qid in qrel:
        
        if qid not in run:
            continue
        num_ranked_q += 1
        
        # docid_and_score = [(docid, score) for docid, score in run[qid].items()]
        # docid_and_score.sort(key=lambda x: x[1], reverse=True)
        # for i, (docid, _) in enumerate(docid_and_score):
        #     rr = 0.0
        #     if cutoff is None or i < cutoff:
        #         if docid in qrel[qid] and qrel[qid][docid] > 0:
        #             rr = 1.0 / (i + 1)
        #             break
        
        docids = run[qid]
        
        for i, docid in enumerate(docids):
            rr = 0.0
            if cutoff is None or i < cutoff:                
                if docid in qrel[qid]:
                    rr = 1.0 / (i + 1)
                    break
        
        results[qid] = rr
        mrr += rr
    mrr /= num_ranked_q
    results["all"] = mrr
    
    return results["all"]



def eval_recall(qrel, run, cutoff=10):
    """
    Compute Recall@cutoff manually.

    Recall@k for a query  q  =
        (# relevant docs retrieved in the top-k) / (total # relevant docs for q)

    The function returns the **mean** recall over all queries that have at least
    one relevant document *and* appear in the run.
    """
    recall_sum   = 0.0
    num_ranked_q = 0
    results      = {}

    for qid, rel_dict in qrel.items():
        # ─── skip queries that have no relevant docs at all ──────────────
        if not rel_dict:
            continue

        if qid not in run:
            # If the query is missing in the run, treat recall as 0.0
            results[qid] = 0.0
            continue

        # Convert run[qid] to an ordered list of docids
        run_entry = run[qid]
        if isinstance(run_entry, dict):
            # sort by score descending
            docids = [d for d, _ in sorted(run_entry.items(),
                                           key=lambda x: x[1],
                                           reverse=True)]
        else:  # assume already-ordered list / iterable
            docids = list(run_entry)

        # ─── count hits in the first k docs ──────────────────────────────
        k = cutoff if cutoff is not None else len(docids)
        retrieved_k = docids[:k]
        num_hits    = sum(1 for d in retrieved_k if d in rel_dict)

        recall = num_hits / len(rel_dict)          # len(rel_dict) > 0 here
        results[qid] = recall

        recall_sum   += recall
        num_ranked_q += 1

    mean_recall = 0.0 if num_ranked_q == 0 else recall_sum / num_ranked_q
    results["all"] = mean_recall
    return results["all"]
