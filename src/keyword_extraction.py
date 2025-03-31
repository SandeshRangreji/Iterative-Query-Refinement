import os
from collections import defaultdict

from datasets import load_dataset
from search import build_bm25_index, build_sbert_index, search_documents

from rake_nltk import Rake
import yake
from keybert import KeyBERT

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def remove_stopwords(query):
    stop_words = set(stopwords.words('english'))
    # Tokenize the query and remove punctuation
    words = word_tokenize(query)
    words = [word for word in words if not all(char in '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~' for char in word)]
    # Filter out stopwords
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return filtered_words


def extract_keywords_rake(docs, top_k=10):
    r = Rake()
    joined_text = "\n\n".join(docs)
    r.extract_keywords_from_text(joined_text)
    return r.get_ranked_phrases()[:top_k]


def extract_keywords_yake(docs, top_k=10):
    joined_text = "\n\n".join(docs)
    kw_extractor = yake.KeywordExtractor(top=top_k, stopwords=None)
    keywords = kw_extractor.extract_keywords(joined_text)
    return [kw for kw, _ in keywords]


def extract_keywords_keybert(model, docs, query, top_k=10):
    joined_text = "\n\n".join(docs)
    keywords = model.extract_keywords(joined_text, top_n=top_k, use_mmr=True, diversity=0.7, seed_keywords=query, keyphrase_ngram_range=(1, 3))
    return [kw for kw, _ in keywords]


def main():
    # ==== Parameters ====
    top_n_docs = 10  # documents to extract keywords from
    top_k_keywords = 10  # keywords per query
    keyword_method = "keybert"  # choose from: "rake", "yake", "keybert"

    print("[Loading datasets]")
    corpus_dataset = load_dataset("BeIR/trec-covid", "corpus")["corpus"]
    queries_dataset = load_dataset("BeIR/trec-covid", "queries")["queries"]

    print("[Building indexes]")
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
        model_name="all-mpnet-base-v2",
        batch_size=64,
        cache_path="sbert_index.pt",
        force_reindex=False
    )
    print("SBERT index built.")

    # Initialize KeyBERT if needed
    keybert_model = KeyBERT(model=sbert_model) if keyword_method == "keybert" else None

    for query_item in queries_dataset:
        query_id = int(query_item["_id"])
        query_text = query_item["text"]
        print(f"\n[Query ID: {query_id}] {query_text}")

        # Hybrid search
        results = search_documents(
            query=query_text,
            bm25=bm25,
            corpus_texts=corpus_texts,
            corpus_ids=corpus_ids,
            sbert_model=sbert_model,
            doc_embeddings=doc_embeddings,
            top_k=top_n_docs,
            method="hybrid",
            use_mmr=False
        )

        # Get top documents
        top_docs = [corpus_texts[corpus_ids.index(doc_id)] for doc_id, _ in results]

        # Keyword extraction
        if keyword_method == "rake":
            keywords = extract_keywords_rake(top_docs, top_k_keywords)
        elif keyword_method == "yake":
            keywords = extract_keywords_yake(top_docs, top_k_keywords)
        elif keyword_method == "keybert":
            seed_keywords = remove_stopwords(query_text)
            print(f"[Seed]: {seed_keywords}")
            keywords = extract_keywords_keybert(keybert_model, top_docs, seed_keywords, top_k_keywords)
        else:
            raise ValueError("Invalid keyword method.")

        print("Top Keywords:")
        for kw in keywords:
            print("-", kw)


if __name__ == "__main__":
    main()