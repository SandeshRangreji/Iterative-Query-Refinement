import math
from collections import defaultdict, Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import download

download("punkt")
download("stopwords")
stop_words = set(stopwords.words("english"))

def tokenize(text):
    tokens = word_tokenize(text.lower())
    return [w for w in tokens if w.isalnum() and w not in stop_words]

def build_local_cooccurrence_matrix(docs, window_size=5):
    word_counts = Counter()
    cooccur_counts = defaultdict(Counter)
    
    for doc in docs:
        tokens = tokenize(doc)
        word_counts.update(tokens)
        
        for i in range(len(tokens)):
            window = tokens[i+1:i+1+window_size]
            for word2 in window:
                word1 = tokens[i]
                if word1 != word2:
                    cooccur_counts[word1][word2] += 1
                    
    total_tokens = sum(word_counts.values())
    return word_counts, cooccur_counts, total_tokens

def compute_pmi(word_counts, cooccur_counts, total_tokens, min_count=2):
    pmi_scores = defaultdict(dict)
    for w1 in cooccur_counts:
        for w2 in cooccur_counts[w1]:
            if word_counts[w1] >= min_count and word_counts[w2] >= min_count:
                p_xy = cooccur_counts[w1][w2] / total_tokens
                p_x = word_counts[w1] / total_tokens
                p_y = word_counts[w2] / total_tokens
                pmi = math.log(p_xy / (p_x * p_y) + 1e-8)
                if pmi > 0:
                    pmi_scores[w1][w2] = pmi
    return pmi_scores

def extract_keywords_pmi(query, pmi_scores, top_k=10):
    query_tokens = tokenize(query)
    keyword_candidates = Counter()

    for qt in query_tokens:
        if qt in pmi_scores:
            keyword_candidates.update(pmi_scores[qt])

    return [kw for kw, _ in keyword_candidates.most_common(top_k)]

def extract_keywords_second_order_pmi(query, pmi_scores, top_k=10):
    query_tokens = tokenize(query)
    second_order_counts = Counter()

    for qt in query_tokens:
        first_order = pmi_scores.get(qt, {})
        for mid_word in first_order:
            second_order = pmi_scores.get(mid_word, {})
            second_order_counts.update(second_order)

    return [kw for kw, _ in second_order_counts.most_common(top_k)]

# Example usage
if __name__ == "__main__":
    from datasets import load_dataset
    from search import build_bm25_index, build_sbert_index, search_documents, cross_encoder_model

    print("[Loading dataset]")
    corpus_dataset = load_dataset("BeIR/trec-covid", "corpus")["corpus"]
    queries_dataset = load_dataset("BeIR/trec-covid", "queries")["queries"]

    print("[Building index]")
    bm25, corpus_texts, corpus_ids = build_bm25_index(
        corpus_dataset, cache_path="bm25_index.pkl", force_reindex=False
    )
    sbert_model, doc_embeddings = build_sbert_index(
        corpus_texts, model_name="all-mpnet-base-v2", batch_size=64, cache_path="sbert_index.pt", force_reindex=False
    )

    for query_item in queries_dataset.select(range(5)):  # Only first query for demo
        query_id = int(query_item["_id"])
        query_text = query_item["text"]
        print(f"\n[Query ID: {query_id}] {query_text}")

        results = search_documents(
            query=query_text,
            bm25=bm25,
            corpus_texts=corpus_texts,
            corpus_ids=corpus_ids,
            sbert_model=sbert_model,
            doc_embeddings=doc_embeddings,
            top_k=500,
            method="hybrid",
            use_mmr=False,
            use_cross_encoder=False,
            cross_encoder_model=cross_encoder_model,
        )

        top_n = 10
        top_docs = [corpus_texts[corpus_ids.index(doc_id)] for doc_id, _ in results][:top_n]

        # Build PMI from top docs
        word_counts, cooccur_counts, total_tokens = build_local_cooccurrence_matrix(top_docs, window_size=5)
        pmi_scores = compute_pmi(word_counts, cooccur_counts, total_tokens)

        # Extract PMI-based keywords
        keywords_pmi = extract_keywords_pmi(query_text, pmi_scores, top_k=5)
        keywords_so_pmi = extract_keywords_second_order_pmi(query_text, pmi_scores, top_k=5)

        print("Top Keywords (PMI):")
        for kw in keywords_pmi:
            print("-", kw)

        print("Top Keywords (Second-order PMI):")
        for kw in keywords_so_pmi:
            print("-", kw)