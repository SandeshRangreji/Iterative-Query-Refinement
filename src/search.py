from collections import defaultdict
import os
import pickle
import random
import numpy as np
import torch
from rank_bm25 import BM25Okapi
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util

# ===============================
# 1) BUILD BM25 INDEX (with caching)
# ===============================
def build_bm25_index(
    corpus_dataset,
    cache_path="bm25_index.pkl",
    force_reindex=False
):
    """
    Builds (or loads) a BM25Okapi index from the corpus.

    :param corpus_dataset: Hugging Face dataset (a list of dicts with '_id', 'text', etc.)
    :param cache_path: Path to a pickle file where we cache the BM25 index and doc info
    :param force_reindex: If True, rebuild the index even if cache exists
    :return: (bm25, corpus_texts, corpus_ids)
    """
    if (not force_reindex) and os.path.exists(cache_path):
        print(f"[BM25] Loading BM25 index from cache: {cache_path}")
        with open(cache_path, "rb") as f:
            bm25, corpus_texts, corpus_ids = pickle.load(f)
        return bm25, corpus_texts, corpus_ids
    
    print("[BM25] Building index from scratch...")
    corpus_texts = []
    corpus_ids = []
    for doc in corpus_dataset:
        corpus_texts.append(doc["text"])
        corpus_ids.append(doc["_id"])
    
    tokenized_corpus = [text.split() for text in corpus_texts]
    bm25 = BM25Okapi(tokenized_corpus)
    
    # Cache results
    print(f"[BM25] Saving BM25 index to cache: {cache_path}")
    with open(cache_path, "wb") as f:
        pickle.dump((bm25, corpus_texts, corpus_ids), f)
    
    return bm25, corpus_texts, corpus_ids

# ===============================
# 2) BUILD SBERT INDEX (with caching)
# ===============================
def build_sbert_index(
    corpus_texts,
    model_name="distilbert-base-nli-stsb-mean-tokens",
    batch_size=64,
    cache_path="sbert_index.pt",
    force_reindex=False
):
    """
    Encodes all documents using a SentenceTransformer model (optionally loads a cache).

    :param corpus_texts: list of document texts
    :param model_name: name of the SBERT model
    :param batch_size: encode in batches
    :param cache_path: Path to a .pt file where we cache doc_embeddings
    :param force_reindex: If True, rebuild embeddings even if cache exists
    :return: (model, doc_embeddings)
    """
    print("[SBERT] Initializing model...")
    model = SentenceTransformer(model_name)

    if (not force_reindex) and os.path.exists(cache_path):
        print(f"[SBERT] Loading SBERT embeddings from cache: {cache_path}")
        data = torch.load(cache_path)
        doc_embeddings = data["doc_embeddings"]
        return model, doc_embeddings
    
    print("[SBERT] Encoding corpus from scratch...")
    doc_embeddings = model.encode(
        corpus_texts, 
        batch_size=batch_size, 
        convert_to_tensor=True
    )
    # Ensure embeddings are on CPU and contiguous
    doc_embeddings = doc_embeddings.cpu().contiguous()

    # Cache the embeddings
    print(f"[SBERT] Saving SBERT embeddings to cache: {cache_path}")
    torch.save({"doc_embeddings": doc_embeddings}, cache_path)
    
    return model, doc_embeddings

# ===============================
# 3) MMR RERANKING
# ===============================
def mmr_rerank(
    query_embedding, 
    doc_embeddings, 
    doc_ids, 
    top_k=10, 
    lambda_param=0.5
):
    """
    Perform Maximal Marginal Relevance (MMR) on the top retrieved documents
    to diversify the results.
    """
    # Compute cosine similarity (N,)
    cos_sim = util.cos_sim(query_embedding, doc_embeddings)[0]  # shape = [N]
    cos_sim = cos_sim.cpu().numpy()

    selected_indices = []
    candidate_indices = list(range(len(doc_ids)))

    for _ in range(min(top_k, len(candidate_indices))):
        mmr_scores = []
        for idx in candidate_indices:
            relevance = cos_sim[idx]
            if len(selected_indices) == 0:
                diversity = 0
            else:
                # compute similarity to all chosen docs, take max
                sim_to_selected = util.cos_sim(
                    doc_embeddings[idx], 
                    doc_embeddings[selected_indices]
                )
                max_sim_to_selected = torch.max(sim_to_selected).item()
                diversity = max_sim_to_selected
            
            mmr_score = lambda_param * relevance - (1 - lambda_param) * diversity
            mmr_scores.append((idx, mmr_score))
        
        # Sort by MMR score descending
        mmr_scores.sort(key=lambda x: x[1], reverse=True)
        
        best_idx = mmr_scores[0][0]
        selected_indices.append(best_idx)
        candidate_indices.remove(best_idx)

    results = [(doc_ids[idx], cos_sim[idx]) for idx in selected_indices]
    return results

# ===============================
# 4) SEARCH FUNCTION
# ===============================
def search_documents(
    query,
    bm25,
    corpus_texts,
    corpus_ids,
    sbert_model=None,
    doc_embeddings=None,
    top_k=10,
    method="bm25",
    use_mmr=False,
    mmr_lambda=0.5
):
    """
    Main search function that supports:
      - BM25-based search
      - SBERT-based semantic search
      - Hybrid search
      - Optional MMR
    """
    results = []

    # =========== BM25 =============
    if method.lower() == "bm25":
        tokenized_query = query.split()
        scores = bm25.get_scores(tokenized_query)
        # Sort by BM25 descending
        sorted_indices = np.argsort(scores)[::-1][:top_k]
        for idx in sorted_indices:
            results.append((corpus_ids[idx], scores[idx]))
        
        if use_mmr and sbert_model is not None and doc_embeddings is not None:
            # MMR re-rank the top candidate docs
            candidate_indices = sorted_indices.tolist()
            candidate_embeddings = doc_embeddings[candidate_indices]  # Torch indexing
            query_embedding = sbert_model.encode(query, convert_to_tensor=True)
            doc_id_subset = [corpus_ids[i] for i in candidate_indices]
            mmr_results = mmr_rerank(
                query_embedding, 
                candidate_embeddings, 
                doc_id_subset, 
                top_k=top_k, 
                lambda_param=mmr_lambda
            )
            return mmr_results
        else:
            return results

    # =========== SBERT =============
    elif method.lower() == "sbert":
        if not sbert_model or doc_embeddings is None:
            raise ValueError("SBERT model and doc_embeddings are required for method='sbert'")

        query_embedding = sbert_model.encode(query, convert_to_tensor=True).cpu()
        cos_scores = util.cos_sim(query_embedding, doc_embeddings)[0].cpu().numpy()
        sorted_indices = np.argsort(cos_scores)[::-1][:top_k]
        for idx in sorted_indices:
            results.append((corpus_ids[idx], cos_scores[idx]))

        if use_mmr:
            candidate_indices = sorted_indices.tolist()
            candidate_embeddings = doc_embeddings[candidate_indices]
            mmr_results = mmr_rerank(
                query_embedding, 
                candidate_embeddings,
                [corpus_ids[i] for i in candidate_indices],
                top_k=top_k,
                lambda_param=mmr_lambda
            )
            return mmr_results
        else:
            return results

    # =========== HYBRID =============
    elif method.lower() == "hybrid":
        if not sbert_model or doc_embeddings is None:
            raise ValueError("SBERT model and doc_embeddings are required for method='hybrid'")
        
        # 1) BM25 top_k
        tokenized_query = query.split()
        bm25_scores = bm25.get_scores(tokenized_query)
        bm25_sorted_indices = np.argsort(bm25_scores)[::-1][:top_k]
        
        # 2) SBERT top_k
        query_embedding = sbert_model.encode(query, convert_to_tensor=True).cpu()
        cos_scores = util.cos_sim(query_embedding, doc_embeddings)[0].cpu().numpy()
        sbert_sorted_indices = np.argsort(cos_scores)[::-1][:top_k]

        # Combine sets
        combined_indices = list(set(bm25_sorted_indices) | set(sbert_sorted_indices))
        bm25_max = max(bm25_scores[combined_indices]) if len(combined_indices) > 0 else 1e-9
        cos_max = max(cos_scores[combined_indices]) if len(combined_indices) > 0 else 1e-9

        doc_final_scores = {}
        for idx in combined_indices:
            # normalize BM25
            bm25_score = bm25_scores[idx] / (bm25_max + 1e-9)
            # normalize cos
            cos_score = cos_scores[idx] / (cos_max + 1e-9)
            # simple sum
            final_score = bm25_score + cos_score
            doc_final_scores[idx] = final_score

        # sort combined by final_score
        sorted_indices = sorted(
            doc_final_scores.keys(), 
            key=lambda x: doc_final_scores[x], 
            reverse=True
        )[:top_k]

        results = [(corpus_ids[i], doc_final_scores[i]) for i in sorted_indices]

        if use_mmr:
            candidate_indices = list(sorted_indices)
            candidate_embeddings = doc_embeddings[candidate_indices]
            mmr_results = mmr_rerank(
                query_embedding, 
                candidate_embeddings,
                [corpus_ids[i] for i in candidate_indices],
                top_k=top_k,
                lambda_param=mmr_lambda
            )
            return mmr_results
        else:
            return results

    else:
        raise ValueError(f"Unknown method: {method}. Use 'bm25', 'sbert', or 'hybrid'.")


def build_qrels_dict(qrels_dataset):
    """
    Convert qrels_dataset into a dictionary of form:
      relevant_docs_by_query[query_id] = set of relevant doc_ids
    Only includes docs with score > 0 (relevant).
    """
    relevant_docs_by_query = defaultdict(set)
    for qid, cid, score in zip(qrels_dataset["query-id"],
                               qrels_dataset["corpus-id"],
                               qrels_dataset["score"]):
        # TREC-COVID: a score > 0 indicates relevance
        if score > 0:
            relevant_docs_by_query[qid].add(cid)
    return relevant_docs_by_query


def evaluate_retrieval_on_queries(
    queries_dataset,
    qrels_dataset,
    bm25,
    corpus_texts,
    corpus_ids,
    sbert_model,
    doc_embeddings,
    method="bm25",
    top_k=10,
    use_mmr=False,
    mmr_lambda=0.5
):
    """
    Runs retrieval on all queries, computes average Precision and Recall.
    Only queries that appear in qrels are evaluated.

    :param queries_dataset: HF dataset with fields "_id" and "text" for queries
    :param qrels_dataset: HF dataset with "query-id", "corpus-id", "score"
    :param bm25: The BM25Okapi object
    :param corpus_texts: list of doc texts (same order as corpus_ids)
    :param corpus_ids: list of doc IDs (same order as corpus_texts)
    :param sbert_model: SentenceTransformer model
    :param doc_embeddings: SBERT doc embeddings
    :param method: "bm25", "sbert", or "hybrid"
    :param top_k: number of docs to retrieve per query
    :param use_mmr: whether to apply MMR
    :param mmr_lambda: MMR lambda param
    :return: (avg_precision, avg_recall, num_evaluated_queries)
    """
    # 1) Build dictionary of relevant docs
    relevant_docs_by_query = build_qrels_dict(qrels_dataset)

    all_precisions = []
    all_recalls = []
    num_evaluated = 0

    # 2) Iterate over each query in queries_dataset
    for query_item in queries_dataset:
        query_id = int(query_item["_id"])  # TREC-COVID query ID
        query_text = query_item["text"]

        # Only evaluate if this query ID is in the Qrels
        if query_id not in relevant_docs_by_query:
            continue

        # Retrieve top documents
        results = search_documents(
            query=query_text,
            bm25=bm25,
            corpus_texts=corpus_texts,
            corpus_ids=corpus_ids,
            sbert_model=sbert_model,
            doc_embeddings=doc_embeddings,
            top_k=top_k,
            method=method,
            use_mmr=use_mmr,
            mmr_lambda=mmr_lambda
        )
        # 'results' is a list of (doc_id, score)
        retrieved_docs = [doc_id for doc_id, _score in results]

        # 3) Compute precision & recall
        relevant_set = relevant_docs_by_query[query_id]
        retrieved_set = set(retrieved_docs)

        num_relevant_retrieved = len(relevant_set.intersection(retrieved_set))
        precision = num_relevant_retrieved / len(retrieved_docs) if len(retrieved_docs) > 0 else 0.0
        recall = num_relevant_retrieved / len(relevant_set) if len(relevant_set) > 0 else 0.0

        all_precisions.append(precision)
        all_recalls.append(recall)
        num_evaluated += 1

    if num_evaluated == 0:
        return 0.0, 0.0, 0

    avg_precision = sum(all_precisions) / num_evaluated
    avg_recall = sum(all_recalls) / num_evaluated
    return avg_precision, avg_recall, num_evaluated

# ===============================
# 5) DEMO in main
# ===============================
if __name__ == "__main__":
    # (Same dataset loading & index building as before)
    print("Loading datasets...")
    corpus_dataset = load_dataset("BeIR/trec-covid", "corpus")["corpus"]
    queries_dataset = load_dataset("BeIR/trec-covid", "queries")["queries"]
    qrels_dataset = load_dataset("BeIR/trec-covid-qrels", split="test")
    print("Datasets loaded.")

    print("Building BM25 index...")
    bm25, corpus_texts, corpus_ids = build_bm25_index(
        corpus_dataset,
        cache_path="bm25_index.pkl",
        force_reindex=False
    )
    print("BM25 index built.")

    print("Building SBERT index...")
    sbert_model, doc_embeddings = build_sbert_index(
        corpus_texts,
        model_name="all-MiniLM-L6-v2",
        batch_size=64,
        cache_path="sbert_index.pt",
        force_reindex=False
    )
    print("SBERT index built.")

    # Evaluate BM25
    bm25_precision, bm25_recall, bm25_count = evaluate_retrieval_on_queries(
        queries_dataset=queries_dataset,
        qrels_dataset=qrels_dataset,
        bm25=bm25,
        corpus_texts=corpus_texts,
        corpus_ids=corpus_ids,
        sbert_model=None,       # Not needed for pure BM25
        doc_embeddings=None,    # Not needed for pure BM25
        method="bm25",
        top_k=500,
        use_mmr=False
    )
    print(f"[BM25] Queries Evaluated: {bm25_count}, Avg Precision: {bm25_precision:.4f}, Avg Recall: {bm25_recall:.4f}")

    # Evaluate SBERT (with or without MMR)
    sbert_precision, sbert_recall, sbert_count = evaluate_retrieval_on_queries(
        queries_dataset=queries_dataset,
        qrels_dataset=qrels_dataset,
        bm25=None,  # not used for SBERT
        corpus_texts=corpus_texts,
        corpus_ids=corpus_ids,
        sbert_model=sbert_model,
        doc_embeddings=doc_embeddings,
        method="sbert",
        top_k=500,
        use_mmr=False,     # Evaluate with MMR
        mmr_lambda=0.7
    )
    print(f"[SBERT] Queries Evaluated: {sbert_count}, Avg Precision: {sbert_precision:.4f}, Avg Recall: {sbert_recall:.4f}")

    # Evaluate SBERT (with or without MMR)
    sbert_mmr_precision, sbert_mmr_recall, sbert_mmr_count = evaluate_retrieval_on_queries(
        queries_dataset=queries_dataset,
        qrels_dataset=qrels_dataset,
        bm25=None,  # not used for SBERT
        corpus_texts=corpus_texts,
        corpus_ids=corpus_ids,
        sbert_model=sbert_model,
        doc_embeddings=doc_embeddings,
        method="sbert",
        top_k=500,
        use_mmr=True,     # Evaluate with MMR
        mmr_lambda=0.7
    )
    print(f"[SBERT + MMR] Queries Evaluated: {sbert_mmr_count}, Avg Precision: {sbert_mmr_precision:.4f}, Avg Recall: {sbert_mmr_recall:.4f}")

    # Evaluate Hybrid (BM25 + SBERT) with MMR
    hybrid_mmr_precision, hybrid_mmr_recall, hybrid_mmr_count = evaluate_retrieval_on_queries(
        queries_dataset=queries_dataset,
        qrels_dataset=qrels_dataset,
        bm25=bm25,
        corpus_texts=corpus_texts,
        corpus_ids=corpus_ids,
        sbert_model=sbert_model,
        doc_embeddings=doc_embeddings,
        method="hybrid",
        top_k=500,
        use_mmr=True,
        mmr_lambda=0.7
    )
    print(f"[Hybrid + MMR] Queries Evaluated: {hybrid_mmr_count}, Avg Precision: {hybrid_mmr_precision:.4f}, Avg Recall: {hybrid_mmr_recall:.4f}")

    # Evaluate Hybrid (BM25 + SBERT) with MMR
    hybrid_precision, hybrid_recall, hybrid_count = evaluate_retrieval_on_queries(
        queries_dataset=queries_dataset,
        qrels_dataset=qrels_dataset,
        bm25=bm25,
        corpus_texts=corpus_texts,
        corpus_ids=corpus_ids,
        sbert_model=sbert_model,
        doc_embeddings=doc_embeddings,
        method="hybrid",
        top_k=500,
        use_mmr=False,
        mmr_lambda=0.7
    )
    print(f"[Hybrid] Queries Evaluated: {hybrid_count}, Avg Precision: {hybrid_precision:.4f}, Avg Recall: {hybrid_recall:.4f}")