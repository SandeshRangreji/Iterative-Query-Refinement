import random
import numpy as np
import torch
from rank_bm25 import BM25Okapi
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util

# -------------------------------
# 1) BUILD BM25 INDEX
# -------------------------------
def build_bm25_index(corpus_dataset):
    """
    Builds a BM25Okapi index from the corpus.

    :param corpus_dataset: Hugging Face dataset (a list of dicts with '_id', 'text', etc.)
    :return: (bm25, corpus_texts, corpus_ids)
    """
    # Extract texts and IDs
    corpus_texts = []
    corpus_ids = []
    for doc in corpus_dataset:
        corpus_texts.append(doc["text"])
        corpus_ids.append(doc["_id"])
    
    # Tokenize
    tokenized_corpus = [text.split() for text in corpus_texts]
    bm25 = BM25Okapi(tokenized_corpus)
    
    return bm25, corpus_texts, corpus_ids

# -------------------------------
# 2) BUILD SBERT INDEX
# -------------------------------
def build_sbert_index(corpus_texts, model_name="distilbert-base-nli-stsb-mean-tokens", batch_size=64):
    """
    Encodes all documents using a Sentence Transformer model.

    :param corpus_texts: list of document texts
    :param model_name: name of the SBERT model (Hugging Face or Sentence-Transformers hub)
    :param batch_size: encode in batches to handle larger corpora
    :return: (model, doc_embeddings)
    """
    model = SentenceTransformer(model_name)
    doc_embeddings = model.encode(corpus_texts, batch_size=batch_size, convert_to_tensor=True)
    
    return model, doc_embeddings

# -------------------------------
# 3) MMR RERANKING
# -------------------------------
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

    :param query_embedding: tensor of shape (768,) – SBERT embedding of query
    :param doc_embeddings: tensor of shape (N, 768) – embeddings of the candidate docs
    :param doc_ids: list of document IDs, length N
    :param top_k: how many documents to return after MMR
    :param lambda_param: trade-off parameter between query relevance and diversity
    :return: list of (doc_id, mmr_score) sorted in MMR order
    """

    # Compute similarity of each doc to the query
    # cos_sim shape: (N,) 
    cos_sim = util.cos_sim(query_embedding, doc_embeddings)[0]  # shape = [N]
    cos_sim = cos_sim.cpu().numpy()

    # We'll keep track of selected indices
    selected_indices = []
    candidate_indices = list(range(len(doc_ids)))

    for _ in range(min(top_k, len(candidate_indices))):
        # For each candidate doc, compute MMR score
        mmr_scores = []
        for idx in candidate_indices:
            # Relevance to query
            relevance = cos_sim[idx]
            
            # Max similarity to already selected docs
            if len(selected_indices) == 0:
                diversity = 0
            else:
                # doc_embeddings[selected_indices] => shape (len(selected_indices), 768)
                # compute similarity to all chosen docs, take max
                sim_to_selected = util.cos_sim(doc_embeddings[idx], doc_embeddings[selected_indices])
                max_sim_to_selected = torch.max(sim_to_selected).item()
                diversity = max_sim_to_selected
            
            mmr_score = lambda_param * relevance - (1 - lambda_param) * diversity
            mmr_scores.append((idx, mmr_score))
        
        # Sort candidates by their MMR score
        mmr_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Pick the best candidate
        best_idx = mmr_scores[0][0]
        selected_indices.append(best_idx)
        # Remove it from candidate list
        candidate_indices.remove(best_idx)

    # Return the selected docs in the order they were chosen
    results = [(doc_ids[idx], cos_sim[idx]) for idx in selected_indices]
    return results

# -------------------------------
# 4) SEARCH FUNCTION
# -------------------------------
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
      - Option to apply MMR re-ranking if doc_embeddings is given

    :param query: user query string
    :param bm25: BM25Okapi object
    :param corpus_texts: list of document texts
    :param corpus_ids: list of document ids
    :param sbert_model: SentenceTransformer model (optional)
    :param doc_embeddings: tensor of shape (N, embedding_dim)
    :param top_k: number of documents to retrieve
    :param method: "bm25" or "sbert" or "hybrid" etc.
    :param use_mmr: boolean, apply MMR for final ranking
    :param mmr_lambda: lambda for MMR (trade-off param)
    :return: list of tuples (doc_id, score) for the top results
    """

    results = []

    if method.lower() == "bm25":
        # BM25 search
        tokenized_query = query.split()
        scores = bm25.get_scores(tokenized_query)
        # Sort by BM25 score descending
        sorted_indices = np.argsort(scores)[::-1][:top_k]
        for idx in sorted_indices:
            results.append((corpus_ids[idx], scores[idx]))

        if use_mmr and sbert_model is not None and doc_embeddings is not None:
            # Re-rank the top X docs by MMR
            # 1) slice out the top candidate embeddings
            candidate_indices = sorted_indices
            candidate_embeddings = doc_embeddings[candidate_indices]
            # 2) embed the query
            query_embedding = sbert_model.encode(query, convert_to_tensor=True)
            # 3) MMR re-rank
            doc_id_subset = [corpus_ids[i] for i in candidate_indices]
            mmr_results = mmr_rerank(query_embedding, candidate_embeddings, doc_id_subset, top_k=top_k, lambda_param=mmr_lambda)
            return mmr_results
        else:
            return results

    elif method.lower() == "sbert":
        # Semantic search with SBERT
        if not sbert_model or doc_embeddings is None:
            raise ValueError("SBERT model and doc embeddings are required for method='sbert'")

        # Encode query
        query_embedding = sbert_model.encode(query, convert_to_tensor=True)
        # Compute cosine similarities
        cos_scores = util.cos_sim(query_embedding, doc_embeddings)[0]  # shape = [N]
        cos_scores = cos_scores.cpu().numpy()
        # Get top_k
        sorted_indices = np.argsort(cos_scores)[::-1][:top_k]
        for idx in sorted_indices:
            results.append((corpus_ids[idx], cos_scores[idx]))

        if use_mmr:
            # MMR re-rank
            candidate_indices = sorted_indices
            candidate_embeddings = doc_embeddings[candidate_indices]
            mmr_results = mmr_rerank(query_embedding, candidate_embeddings,
                                     [corpus_ids[i] for i in candidate_indices],
                                     top_k=top_k, lambda_param=mmr_lambda)
            return mmr_results
        else:
            return results

    elif method.lower() == "hybrid":
        # Example simple hybrid: get top_k from BM25 + top_k from SBERT, then unify
        # (You could do more sophisticated weighting or re-ranking.)
        
        # BM25
        tokenized_query = query.split()
        bm25_scores = bm25.get_scores(tokenized_query)
        bm25_sorted_indices = np.argsort(bm25_scores)[::-1][:top_k]
        
        # SBERT
        if not sbert_model or doc_embeddings is None:
            raise ValueError("SBERT model and doc embeddings are required for method='hybrid'")
        query_embedding = sbert_model.encode(query, convert_to_tensor=True)
        cos_scores = util.cos_sim(query_embedding, doc_embeddings)[0].cpu().numpy()
        sbert_sorted_indices = np.argsort(cos_scores)[::-1][:top_k]

        # Combine
        combined_indices = list(set(bm25_sorted_indices) | set(sbert_sorted_indices))
        # Just for demonstration, we sum normalized scores from BM25 and SBERT:
        # Normalizing scores
        bm25_max = max(bm25_scores[combined_indices])
        cos_max = max(cos_scores[combined_indices]) if len(combined_indices) > 0 else 1.0
        
        doc_final_scores = {}
        for idx in combined_indices:
            # normalize BM25 score
            bm25_score = bm25_scores[idx] / (bm25_max + 1e-9)
            # normalize cos score
            cos_score = cos_scores[idx] / (cos_max + 1e-9)
            # sum them (simple approach)
            final_score = bm25_score + cos_score
            doc_final_scores[idx] = final_score

        # sort by final score
        sorted_indices = sorted(doc_final_scores.keys(), key=lambda x: doc_final_scores[x], reverse=True)[:top_k]
        
        results = [(corpus_ids[i], doc_final_scores[i]) for i in sorted_indices]

        if use_mmr:
            # MMR re-rank
            candidate_indices = sorted_indices
            candidate_embeddings = doc_embeddings[candidate_indices]
            mmr_results = mmr_rerank(query_embedding, candidate_embeddings,
                                     [corpus_ids[i] for i in candidate_indices],
                                     top_k=top_k,
                                     lambda_param=mmr_lambda)
            return mmr_results
        else:
            return results

    else:
        raise ValueError(f"Unknown method: {method}. Use 'bm25', 'sbert', or 'hybrid'.")


# -------------------------------
# 5) PUTTING IT ALL TOGETHER
# -------------------------------
if __name__ == "__main__":
    # Example usage, assuming you have already loaded the dataset as in your snippet:
    print("Loading datasets...")
    # --------------------------------------------------------------------------
    corpus_dataset = load_dataset("BeIR/trec-covid", "corpus")["corpus"]
    queries_dataset = load_dataset("BeIR/trec-covid", "queries")["queries"]
    qrels_dataset = load_dataset("BeIR/trec-covid-qrels", split="test")
    # --------------------------------------------------------------------------
    print("Datasets loaded.")

    # Step A: Build BM25 index
    print("Building BM25 index...")
    bm25, corpus_texts, corpus_ids = build_bm25_index(corpus_dataset)
    print("BM25 index built.")

    # Step B: Build SBERT index
    print("Building SBERT index...")
    sbert_model, doc_embeddings = build_sbert_index(corpus_texts, model_name="distilbert-base-nli-stsb-mean-tokens")
    print("SBERT index built.")

    # Let's say we have a user query:
    example_query = "how does covid-19 spread"
    
    # Step C: Retrieve with BM25
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
        use_mmr=False  # or True if you want MMR
    )
    print("Retrieval done.")

    # Step D: Retrieve with SBERT
    print("Running retrieval with SBERT...")
    results_sbert = search_documents(
        query=example_query,
        bm25=None,  # not needed for sbert
        corpus_texts=corpus_texts,
        corpus_ids=corpus_ids,
        sbert_model=sbert_model,
        doc_embeddings=doc_embeddings,
        top_k=10,
        method="sbert",
        use_mmr=True,      # Turn MMR on if you want diversity
        mmr_lambda=0.7     # Can tweak this value
    )
    print("Retrieval done.")

    # Step E: Hybrid retrieval (simple approach)
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
        use_mmr=True,    # MMR on hybrid
        mmr_lambda=0.5
    )
    print("Retrieval done.")

    print("BM25 results:", results_bm25)
    print("SBERT results:", results_sbert)
    print("Hybrid results:", results_hybrid)

    print("Step 1 code for retrieval is ready. Adjust usage in main as needed.")