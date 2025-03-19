import random
import numpy as np
from datasets import load_dataset
from collections import Counter

# Load datasets from Hugging Face
corpus_dataset = load_dataset("BeIR/trec-covid", "corpus")["corpus"]
queries_dataset = load_dataset("BeIR/trec-covid", "queries")["queries"]
qrels_dataset = load_dataset("BeIR/trec-covid-qrels", split="test")

# Print dataset information
num_documents = len(corpus_dataset)
num_queries = len(queries_dataset)
total_judgments = len(qrels_dataset)

print(f"Total number of documents: {num_documents}")
print(f"Total number of queries: {num_queries}")
print(f"Total number of query-document relevance judgments: {total_judgments}")

# Randomly print one document
random_doc_idx = random.randrange(len(corpus_dataset))
random_doc = corpus_dataset[random_doc_idx]
print("\nRandom Document Sample:")
print(f"Document ID: {random_doc['_id']}")
print(f"Title: {random_doc['title']}")
print(f"Text (start): {random_doc['text']}...\n")  # Show first 200 characters

# Compute query/document length statistics
query_lengths = [len(query["text"].split()) for query in queries_dataset]
doc_lengths = [len(doc["text"].split()) for doc in corpus_dataset]

avg_query_length = np.mean(query_lengths)
median_query_length = np.median(query_lengths)
std_query_length = np.std(query_lengths)

avg_doc_length = np.mean(doc_lengths)
median_doc_length = np.median(doc_lengths)
std_doc_length = np.std(doc_lengths)

min_query_len = min(query_lengths)
max_query_len = max(query_lengths)
min_doc_len = min(doc_lengths)
max_doc_len = max(doc_lengths)

# Most frequent query and document lengths
query_length_counts = Counter(query_lengths).most_common(5)
doc_length_counts = Counter(doc_lengths).most_common(5)

# Compute relevance statistics
query_counts = Counter(qrels_dataset["query-id"])
doc_counts = Counter(qrels_dataset["corpus-id"])
score_counts = Counter(qrels_dataset["score"])

avg_judgments_per_query = total_judgments / num_queries
total_relevant_judgments = sum(1 for s in qrels_dataset["score"] if s > 0)
avg_relevant_per_query = total_relevant_judgments / num_queries

# Unique queries and documents in qrels
unique_queries_in_qrels = len(set(qrels_dataset["query-id"]))
unique_docs_in_qrels = len(set(qrels_dataset["corpus-id"]))

# Number of relevance judgments per query
judgments_per_query = list(query_counts.values())

min_judgments_per_query = min(judgments_per_query)
max_judgments_per_query = max(judgments_per_query)
median_judgments_per_query = np.median(judgments_per_query)

# Number of relevance judgments per document
judgments_per_doc = list(doc_counts.values())

min_judgments_per_doc = min(judgments_per_doc)
max_judgments_per_doc = max(judgments_per_doc)
median_judgments_per_doc = np.median(judgments_per_doc)

# Compute number of relevant documents per query correctly
relevant_docs_per_query = Counter()

# Iterate over qrels_dataset to count relevant documents per query
for query_id, score in zip(qrels_dataset["query-id"], qrels_dataset["score"]):
    if score > 0:  # Count only relevant documents (score > 0)
        relevant_docs_per_query[query_id] += 1

# Convert counts to a list for statistics
relevant_docs_counts = list(relevant_docs_per_query.values())

# Compute percentiles
percentiles = np.percentile(relevant_docs_counts, [25, 50, 75, 90]) if relevant_docs_counts else [0, 0, 0, 0]

# Percentage of queries with at least one relevant document
queries_with_relevant_docs = sum(1 for count in relevant_docs_counts if count > 0)
percent_queries_with_relevant = (queries_with_relevant_docs / num_queries) * 100 if num_queries > 0 else 0

# Print all results
print("\n=================== Query and Document Statistics ===================")
print(f"Average query length (words): {avg_query_length:.2f}")
print(f"Median query length (words): {median_query_length:.2f}")
print(f"Std dev query length: {std_query_length:.2f}")
print(f"Shortest query length (words): {min_query_len}")
print(f"Longest query length (words): {max_query_len}")

print(f"Average document length (words): {avg_doc_length:.2f}")
print(f"Median document length (words): {median_doc_length:.2f}")
print(f"Std dev document length: {std_doc_length:.2f}")
print(f"Shortest document length (words): {min_doc_len}")
print(f"Longest document length (words): {max_doc_len}")

print("\nMost common query lengths (length: count):")
for length, count in query_length_counts:
    print(f"  {length}: {count}")

print("\nMost common document lengths (length: count):")
for length, count in doc_length_counts:
    print(f"  {length}: {count}")

print("\n=================== Relevance Judgment Statistics ===================")
print(f"Total number of query-document relevance judgments: {total_judgments}")
print(f"Average number of judgments per query: {avg_judgments_per_query:.1f}")
print(f"Average number of relevant documents per query: {avg_relevant_per_query:.1f}")

print(f"Total unique queries in qrels: {unique_queries_in_qrels}")
print(f"Total unique documents in qrels: {unique_docs_in_qrels}")

print(f"\nMin/Max/Median judgments per query: {min_judgments_per_query}, {max_judgments_per_query}, {median_judgments_per_query}")
print(f"Min/Max/Median judgments per document: {min_judgments_per_doc}, {max_judgments_per_doc}, {median_judgments_per_doc}")

print(f"\nRelevance score distribution (score: count):")
for score_value in sorted(score_counts.keys()):
    print(f"  {score_value}: {score_counts[score_value]}")

print("\n=================== Relevant Document Distribution ===================")
print("Distribution of relevant documents per query (percentiles):")
print(f"  25th percentile: {percentiles[0]:.1f}")
print(f"  50th percentile (median): {percentiles[1]:.1f}")
print(f"  75th percentile: {percentiles[2]:.1f}")
print(f"  90th percentile: {percentiles[3]:.1f}")

print(f"\nPercentage of queries with at least one relevant document: {percent_queries_with_relevant:.2f}%")