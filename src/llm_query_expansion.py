import pandas as pd
from collections import defaultdict
from datasets import load_dataset
from search import build_bm25_index, build_sbert_index, search_documents, cross_encoder_model
from sentence_transformers import SentenceTransformer

def reciprocal_rank_fusion(rankings, k=60):
    scores = defaultdict(float)
    for rank_list in rankings:
        for rank, (doc_id, _) in enumerate(rank_list):
            scores[doc_id] += 1 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

def evaluate_precomputed_results(query_results, qrels_dataset, top_k_p=20, top_k_r=1000):
    from search import build_qrels_dicts

    relevant_docs_by_query, highly_relevant_docs_by_query, overall_relevant_docs_by_query = build_qrels_dicts(qrels_dataset)

    all_precisions = {'relevant': [], 'highly_relevant': [], 'overall': []}
    all_recalls = {'relevant': [], 'highly_relevant': [], 'overall': []}
    num_evaluated = 0

    for q in query_results:
        qid = q["query_id"]
        results = q["results"]
        retrieved_docs = [doc_id for doc_id, _ in results[:top_k_p]]
        full_retrieved_docs = [doc_id for doc_id, _ in results[:top_k_r]]

        def calc_prec_recall(relevant_set):
            if not relevant_set:
                return None, None
            retrieved_set = set(retrieved_docs)
            top_k_r_set = set(full_retrieved_docs)
            num_relevant_retrieved = len(relevant_set.intersection(retrieved_set))
            num_relevant_in_top_r = len(relevant_set.intersection(top_k_r_set))
            precision = num_relevant_retrieved / top_k_p
            recall = num_relevant_in_top_r / len(relevant_set)
            return precision, recall

        for level, qrels in [
            ("relevant", relevant_docs_by_query),
            ("highly_relevant", highly_relevant_docs_by_query),
            ("overall", overall_relevant_docs_by_query),
        ]:
            if qid in qrels:
                p, r = calc_prec_recall(qrels[qid])
                if p is not None:
                    all_precisions[level].append(p)
                    all_recalls[level].append(r)

        num_evaluated += 1

    avg_precisions = {lvl: sum(vals)/len(vals) if vals else 0.0 for lvl, vals in all_precisions.items()}
    avg_recalls = {lvl: sum(vals)/len(vals) if vals else 0.0 for lvl, vals in all_recalls.items()}
    return avg_precisions, avg_recalls, num_evaluated

def main():
    top_k_results = 1000

    print("[Loading datasets]")
    corpus_dataset = load_dataset("BeIR/trec-covid", "corpus")["corpus"]
    queries_dataset = load_dataset("BeIR/trec-covid", "queries")["queries"]
    qrels_dataset = load_dataset("BeIR/trec-covid-qrels", split="test")

    print("[Building indexes]")
    bm25, corpus_texts, corpus_ids = build_bm25_index(corpus_dataset, cache_path="bm25_index.pkl", force_reindex=False)
    sbert_model, doc_embeddings = build_sbert_index(corpus_texts, model_name="all-mpnet-base-v2", batch_size=64, cache_path="sbert_index.pt", force_reindex=False)

    print("[Reading expanded queries CSV]")
    clean_df = pd.read_csv("data/clean_expanded_queries.csv")  # Must contain 'expanded_query' column

    all_query_results_original = []
    all_query_results_rrf = []

    for idx, row in clean_df.iterrows():
        query_item = queries_dataset[idx]
        query_id = int(query_item["_id"])
        original_query = query_item["text"]
        expanded_queries = row["expanded_query"].split('\n')

        print(f"\n[Query ID: {query_id}] {original_query}")

        original_results = search_documents(
            query=original_query,
            bm25=bm25,
            corpus_texts=corpus_texts,
            corpus_ids=corpus_ids,
            sbert_model=sbert_model,
            doc_embeddings=doc_embeddings,
            top_k=top_k_results,
            method="hybrid",
            use_mmr=False,
            use_cross_encoder=False,
            cross_encoder_model=cross_encoder_model,
        )

        all_query_results_original.append({
            "query_id": query_id,
            "query_text": original_query,
            "results": original_results
        })

        keyword_results = []
        for query in expanded_queries:
            results = search_documents(
                query=query,
                bm25=bm25,
                corpus_texts=corpus_texts,
                corpus_ids=corpus_ids,
                sbert_model=sbert_model,
                doc_embeddings=doc_embeddings,
                top_k=top_k_results,
                method="hybrid",
                use_mmr=False,
                use_cross_encoder=False,
                cross_encoder_model=None,
            )
            keyword_results.append(results)

        all_rankings = [original_results] + keyword_results
        combined_results = reciprocal_rank_fusion(all_rankings)

        all_query_results_rrf.append({
            "query_id": query_id,
            "query_text": original_query,
            "results": combined_results
        })

    print("\n[Evaluating Original Query Only Results]")
    avg_precisions, avg_recalls, num_evaluated = evaluate_precomputed_results(
        query_results=all_query_results_original,
        qrels_dataset=qrels_dataset,
        top_k_p=20,
        top_k_r=1000,
    )
    for level in ['relevant', 'highly_relevant', 'overall']:
        print(f"[Original Query] [{level.capitalize()}] Avg Precision: {avg_precisions[level]:.4f}, Avg Recall: {avg_recalls[level]:.4f}")

    print("\n[Evaluating RRF-based Query Expansion Results]")
    avg_precisions, avg_recalls, num_evaluated = evaluate_precomputed_results(
        query_results=all_query_results_rrf,
        qrels_dataset=qrels_dataset,
        top_k_p=20,
        top_k_r=1000,
    )
    for level in ['relevant', 'highly_relevant', 'overall']:
        print(f"[Query Expansion + RRF] [{level.capitalize()}] Avg Precision: {avg_precisions[level]:.4f}, Avg Recall: {avg_recalls[level]:.4f}")

if __name__ == "__main__":
    main()
