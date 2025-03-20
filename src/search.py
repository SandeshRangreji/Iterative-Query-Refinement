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


# ===============================
# 5) DEMO in main
# ===============================
if __name__ == "__main__":
    print("Loading datasets...")
    corpus_dataset = load_dataset("BeIR/trec-covid", "corpus")["corpus"]
    queries_dataset = load_dataset("BeIR/trec-covid", "queries")["queries"]
    qrels_dataset = load_dataset("BeIR/trec-covid-qrels", split="test")
    print("Datasets loaded.")

    # Step A: Build/Load BM25 index
    bm25_cache_path = "bm25_index.pkl"
    print("Building BM25 index...")
    bm25, corpus_texts, corpus_ids = build_bm25_index(
        corpus_dataset,
        cache_path=bm25_cache_path,
        force_reindex=False  # set True if you want to rebuild
    )
    print("BM25 index built.")

    # Step B: Build/Load SBERT index
    sbert_cache_path = "sbert_index.pt"
    print("Building SBERT index...")
    sbert_model, doc_embeddings = build_sbert_index(
        corpus_texts,
        model_name="distilbert-base-nli-stsb-mean-tokens",
        batch_size=64,
        cache_path=sbert_cache_path,
        force_reindex=False  # set True if you want to rebuild
    )
    print("SBERT index built.")

    example_query = "how does covid-19 spread"

    # Step C: BM25 search
    print("Running retrieval with BM25...")
    results_bm25 = search_documents(
        query=example_query,
        bm25=bm25,
        corpus_texts=corpus_texts,
        corpus_ids=corpus_ids,
        sbert_model=sbert_model,
        doc_embeddings=doc_embeddings,
        top_k=10,
        method="bm25",
        use_mmr=False  # or True for MMR
    )
    print("Retrieval done. Example BM25 results:")
    print(results_bm25)

    # Step D: SBERT search
    print("Running retrieval with SBERT...")
    results_sbert = search_documents(
        query=example_query,
        bm25=None,  # not needed
        corpus_texts=corpus_texts,
        corpus_ids=corpus_ids,
        sbert_model=sbert_model,
        doc_embeddings=doc_embeddings,
        top_k=10,
        method="sbert",
        use_mmr=True,      # Turn MMR on if you want diversity
        mmr_lambda=0.7
    )
    print("Retrieval done. Example SBERT results:")
    print(results_sbert)

    # Step E: Hybrid
    print("Running retrieval with Hybrid (BM25 + SBERT)...")
    results_hybrid = search_documents(
        query=example_query,
        bm25=bm25,
        corpus_texts=corpus_texts,
        corpus_ids=corpus_ids,
        sbert_model=sbert_model,
        doc_embeddings=doc_embeddings,
        top_k=10,
        method="hybrid",
        use_mmr=True,
        mmr_lambda=0.5
    )
    print("Retrieval done. Example Hybrid results:")
    print(results_hybrid)

    print("Step 1 code (with caching and negative-stride fix) is complete.")