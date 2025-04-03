import random
from datasets import load_dataset
from collections import defaultdict

# Load datasets from Hugging Face
corpus_dataset = load_dataset("BeIR/trec-covid", "corpus")["corpus"]
queries_dataset = load_dataset("BeIR/trec-covid", "queries")["queries"]
qrels_dataset = load_dataset("BeIR/trec-covid-qrels", split="test")

# Create dictionaries to map IDs to their texts
query_id_to_text = {query["_id"]: query["text"] for query in queries_dataset}
doc_id_to_text = {doc["_id"]: doc for doc in corpus_dataset}

# Create dictionaries to store relevance info
query_to_relevant_docs = defaultdict(list)
query_to_highly_relevant_docs = defaultdict(list)

# Track all relevance info
for qrel in qrels_dataset:
    score = qrel["score"]
    query_id = qrel["query-id"]
    doc_id = qrel["corpus-id"]

    if score == 1:
        query_to_relevant_docs[query_id].append(doc_id)
    elif score == 2:
        query_to_highly_relevant_docs[query_id].append(doc_id)

# Combine both sets to get overall relevant docs (score > 0)
query_to_all_relevant_docs = defaultdict(set)
for qid in set(query_to_relevant_docs) | set(query_to_highly_relevant_docs):
    rel_set = set(query_to_relevant_docs.get(qid, []))
    high_rel_set = set(query_to_highly_relevant_docs.get(qid, []))
    query_to_all_relevant_docs[qid] = rel_set | high_rel_set

# Select a random sample of queries
num_samples = 5
random_query_ids = random.sample(list(query_to_all_relevant_docs.keys()), num_samples)

# Display the sampled queries and stats
for query_id in random_query_ids:
    query_text = query_id_to_text.get(str(query_id), "N/A")
    relevant_docs = set(query_to_relevant_docs.get(query_id, []))
    highly_relevant_docs = set(query_to_highly_relevant_docs.get(query_id, []))
    all_relevant_docs = query_to_all_relevant_docs.get(query_id, set())

    print(f"\nQuery ID: {query_id}")
    print(f"Query Text: {query_text}")
    print(f"Relevant Document IDs (score=1): {len(relevant_docs)}")
    print(f"Highly Relevant Document IDs (score=2): {len(highly_relevant_docs)}")
    print(f"Overall Relevant Documents (score > 0): {len(all_relevant_docs)}")
    print(f"Intersection (score=1 ∩ score=2): {len(relevant_docs & highly_relevant_docs)}")

    # Sample a document to preview
    if all_relevant_docs:
        sample_doc_id = next(iter(all_relevant_docs))
        sample_doc = doc_id_to_text.get(sample_doc_id, {})
        sample_doc_title = sample_doc.get("title", "No title available")
        sample_doc_text = sample_doc.get("text", "No text available")
        print(f"\nSample Relevant Document ID: {sample_doc_id}")
        print(f"Title: {sample_doc_title}")
        print(f"Text (excerpt): {sample_doc_text[:500]}...\n")