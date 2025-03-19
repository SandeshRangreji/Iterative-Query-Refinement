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

# Create a dictionary to map query IDs to relevant document IDs
query_to_relevant_docs = defaultdict(list)
for qrel in qrels_dataset:
    if qrel["score"] > 0:  # Consider only relevant documents
        query_to_relevant_docs[qrel["query-id"]].append(qrel["corpus-id"])

# Select a random sample of queries
num_samples = 5  # Number of random queries to display
random_query_ids = random.sample(list(query_to_relevant_docs.keys()), num_samples)

# Display the random queries, their relevant document IDs, and a sample document
for query_id in random_query_ids:
    query_text = query_id_to_text.get(str(query_id), "N/A")
    relevant_docs = query_to_relevant_docs[query_id]
    print(f"\nQuery ID: {query_id}")
    print(f"Query Text: {query_text}")
    print(f"Relevant Document IDs: {', '.join(relevant_docs)}")
    print(f"Number of Relevant Documents: {len(relevant_docs)}")

    # Display a sample relevant document
    sample_doc_id = relevant_docs[0] if relevant_docs else None
    sample_doc = doc_id_to_text.get(sample_doc_id, {})
    sample_doc_title = sample_doc.get("title", "No title available")
    sample_doc_text = sample_doc.get("text", "No text available")
    print(f"\nSample Relevant Document ID: {sample_doc_id}")
    print(f"Title: {sample_doc_title}")
    print(f"Text (excerpt): {sample_doc_text[:500]}...\n")  # Display first 500 characters