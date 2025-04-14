import os
from collections import defaultdict, Counter
from datasets import load_dataset
from search import build_bm25_index, build_sbert_index, search_documents, cross_encoder_model
from keyword_extraction import remove_stopwords, extract_keywords_keybert_filtered
from pmi_keyphrases import extract_keywords_pmi, extract_keywords_second_order_pmi
from keybert import KeyBERT
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
    expansion_method = "keybert"  # Options: "keybert", "pmi", "second_order_pmi"

    top_n_docs = 10
    top_k_keywords = 5
    top_k_results = 1000
    original_query_weight = 0.7

    print("[Loading datasets]")
    corpus_dataset = load_dataset("BeIR/trec-covid", "corpus")["corpus"]
    queries_dataset = load_dataset("BeIR/trec-covid", "queries")["queries"]
    qrels_dataset = load_dataset("BeIR/trec-covid-qrels", split="test")

    print("[Building indexes]")
    bm25, corpus_texts, corpus_ids = build_bm25_index(corpus_dataset, cache_path="bm25_index.pkl", force_reindex=False)
    sbert_model, doc_embeddings = build_sbert_index(corpus_texts, model_name="all-mpnet-base-v2", batch_size=64, cache_path="sbert_index.pt", force_reindex=False)

    keybert_model = KeyBERT(model=sbert_model)
    all_query_results_rrf = []
    all_query_results_original = []
    all_query_results_concatenated = []

    for query_item in queries_dataset:
        query_id = int(query_item["_id"])
        query_text = query_item["text"]
        print(f"\n[Query ID: {query_id}] {query_text}")

        # --- Original Search ---
        original_results = search_documents(
            query=query_text,
            bm25=bm25,
            corpus_texts=corpus_texts,
            corpus_ids=corpus_ids,
            sbert_model=sbert_model,
            doc_embeddings=doc_embeddings,
            top_k=top_k_results,
            method="hybrid",
            use_mmr=False,
            use_cross_encoder=True,
            cross_encoder_model=cross_encoder_model,
        )

        all_query_results_original.append({
            "query_id": query_id,
            "query_text": query_text,
            "results": original_results
        })

        top_docs = [corpus_texts[corpus_ids.index(doc_id)] for doc_id, _ in original_results[:top_n_docs]]
        query_embedding = sbert_model.encode(query_text, convert_to_tensor=True)

        if expansion_method == "keybert":
            seed_keywords = remove_stopwords(query_text)
            keywords = extract_keywords_keybert_filtered(
                keybert_model, sbert_model,
                top_docs, seed_keywords,
                top_k=top_k_keywords,
                query_embedding=query_embedding
            )

        elif expansion_method == "pmi":
            keywords = extract_keywords_pmi(query_text, top_docs, top_k=top_k_keywords)

        elif expansion_method == "second_order_pmi":
            keywords = extract_keywords_second_order_pmi(query_text, top_docs, top_k=top_k_keywords)

        else:
            raise ValueError(f"Unsupported expansion method: {expansion_method}")

        # --- RRF-based Expansion ---
        keyword_results = []
        overlap_counter = Counter()

        for kw in keywords:
            kw_results = search_documents(
                query=kw,
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
            keyword_results.append(kw_results)
            overlap = set(doc_id for doc_id, _ in kw_results).intersection(doc_id for doc_id, _ in original_results)
            overlap_counter[kw] = len(overlap)

        print("[Keyword Overlap with Original Query Results]:")
        for kw in keywords:
            print(f" - '{kw}': {overlap_counter[kw]} overlapping docs")

        weighted_original = [(doc_id, score * original_query_weight) for doc_id, score in original_results]
        all_rankings = [weighted_original] + keyword_results
        combined_results = reciprocal_rank_fusion(all_rankings)

        all_query_results_rrf.append({
            "query_id": query_id,
            "query_text": query_text,
            "results": combined_results
        })

        # --- Concatenated Expansion ---
        expanded_query_text = query_text + " " + " ".join(keywords)
        print(f"[Expanded Query Text] {expanded_query_text}")

        expanded_results = search_documents(
            query=expanded_query_text,
            bm25=bm25,
            corpus_texts=corpus_texts,
            corpus_ids=corpus_ids,
            sbert_model=sbert_model,
            doc_embeddings=doc_embeddings,
            top_k=top_k_results,
            method="hybrid",
            use_mmr=False,
            use_cross_encoder=True,
            cross_encoder_model=cross_encoder_model,
        )

        all_query_results_concatenated.append({
            "query_id": query_id,
            "query_text": expanded_query_text,
            "results": expanded_results
        })

    # --- Evaluation ---
    print("\n[Evaluating Original Query Only Results]")
    avg_precisions, avg_recalls, num_evaluated = evaluate_precomputed_results(
        query_results=all_query_results_original,
        qrels_dataset=qrels_dataset,
        top_k_p=20,
        top_k_r=1000,
    )
    print(f"[Original Query] Queries Evaluated: {num_evaluated}")
    for level in ['relevant', 'highly_relevant', 'overall']:
        print(f"[Original Query] [{level.capitalize()}] Avg Precision: {avg_precisions[level]:.4f}, Avg Recall: {avg_recalls[level]:.4f}")

    print("\n[Evaluating Weighted RRF-based Query Expansion]")
    avg_precisions, avg_recalls, num_evaluated = evaluate_precomputed_results(
        query_results=all_query_results_rrf,
        qrels_dataset=qrels_dataset,
        top_k_p=20,
        top_k_r=1000,
    )
    print(f"[Query Expansion + RRF] Queries Evaluated: {num_evaluated}")
    for level in ['relevant', 'highly_relevant', 'overall']:
        print(f"[Query Expansion + RRF] [{level.capitalize()}] Avg Precision: {avg_precisions[level]:.4f}, Avg Recall: {avg_recalls[level]:.4f}")

    print("\n[Evaluating Concatenated Expanded Query Results]")
    avg_precisions, avg_recalls, num_evaluated = evaluate_precomputed_results(
        query_results=all_query_results_concatenated,
        qrels_dataset=qrels_dataset,
        top_k_p=20,
        top_k_r=1000,
    )
    print(f"[Concatenated Expansion] Queries Evaluated: {num_evaluated}")
    for level in ['relevant', 'highly_relevant', 'overall']:
        print(f"[Concatenated Expansion] [{level.capitalize()}] Avg Precision: {avg_precisions[level]:.4f}, Avg Recall: {avg_recalls[level]:.4f}")

    # --- Recall Gain Comparison ---
    original_ids = set(doc_id for doc_id, _ in original_results[:top_k_results])
    expanded_ids = set(doc_id for doc_id, _ in combined_results[:top_k_results])
    concat_ids = set(doc_id for doc_id, _ in expanded_results[:top_k_results])

    print(f"\nRecall gain from RRF Expansion: {len(expanded_ids - original_ids)} new docs added")
    print(f"Recall gain from Concatenated Expansion: {len(concat_ids - original_ids)} new docs added")


if __name__ == "__main__":
    main()