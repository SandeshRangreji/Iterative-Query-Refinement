import os
import math
from collections import defaultdict

from datasets import load_dataset
from search import build_bm25_index, build_sbert_index, search_documents, cross_encoder_model

from rake_nltk import Rake
import yake
from keybert import KeyBERT

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sentence_transformers import util

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

def compute_idf(top_docs_keywords, total_docs):
    df = defaultdict(int)
    for kws in top_docs_keywords:
        for kw in set(kws):
            df[kw] += 1
    idf = {kw: math.log(total_docs / (1 + freq)) for kw, freq in df.items()}
    return idf

def extract_keywords_keybert_filtered(model, sbert_model, docs, query, top_k=10, query_embedding=None):
    # Extract keywords from each document
    # all_keywords = []
    # for doc in docs:
    #     keywords = model.extract_keywords(doc, top_n=20, use_mmr=True, diversity=0.7, seed_keywords=query, keyphrase_ngram_range=(1, 3))
    #     all_keywords.append([kw for kw, _ in keywords])
    original_keywords = extract_keywords_keybert(model, docs, query, top_k=20)
    all_keywords = [original_keywords]

    # Compute IDF over the 10 documents
    idf_scores = compute_idf(all_keywords, len(docs))

    # Filter out low-IDF keywords (threshold = 0.5)
    filtered_keywords = [kw for kw in original_keywords if idf_scores.get(kw, 0.0) > 0.25]

    # Rank by similarity to query
    keyword_list = list(filtered_keywords)
    # keyword_embeddings = sbert_model.encode(keyword_list, convert_to_tensor=True)
    # cos_scores = util.cos_sim(query_embedding, keyword_embeddings)[0]

    # ranked_keywords = sorted(zip(keyword_list, cos_scores.tolist()), key=lambda x: x[1], reverse=True)
    return keyword_list[:top_k]

def main():
    top_n_docs = 10
    top_k_keywords = 5
    keyword_method = "keybert"

    print("[Loading datasets]")
    corpus_dataset = load_dataset("BeIR/trec-covid", "corpus")["corpus"]
    queries_dataset = load_dataset("BeIR/trec-covid", "queries")["queries"]

    print("[Building indexes]")
    bm25, corpus_texts, corpus_ids = build_bm25_index(
        corpus_dataset, cache_path="bm25_index.pkl", force_reindex=False
    )
    sbert_model, doc_embeddings = build_sbert_index(
        corpus_texts, model_name="all-mpnet-base-v2", batch_size=64, cache_path="sbert_index.pt", force_reindex=False
    )

    keybert_model = KeyBERT(model=sbert_model) if keyword_method == "keybert" else None

    for query_item in queries_dataset:
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
            top_k=top_n_docs,
            method="hybrid",
            use_mmr=False,
            use_cross_encoder=False,
            cross_encoder_model=None,
        )

        top_docs = [corpus_texts[corpus_ids.index(doc_id)] for doc_id, _ in results]

        if keyword_method == "rake":
            keywords = extract_keywords_rake(top_docs, top_k_keywords)
        elif keyword_method == "yake":
            keywords = extract_keywords_yake(top_docs, top_k_keywords)
        elif keyword_method == "keybert":
            seed_keywords = remove_stopwords(query_text)
            print(f"[Seed]: {seed_keywords}")
            query_embedding = sbert_model.encode(query_text, convert_to_tensor=True)
            keywords = extract_keywords_keybert_filtered(keybert_model, sbert_model, top_docs, seed_keywords, top_k_keywords, query_embedding=query_embedding)
        else:
            raise ValueError("Invalid keyword method.")

        print("Top Keywords:")
        for kw in keywords:
            print("-", kw)

if __name__ == "__main__":
    main()
