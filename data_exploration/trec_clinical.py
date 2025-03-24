import statistics
from datasets import load_dataset

# Load queries and qrels using the Hugging Face IR-datasets configuration.
queries_dataset = load_dataset("irds/clinicaltrials_2021_trec-ct-2021", "queries", trust_remote_code=True)
qrels_dataset   = load_dataset("irds/clinicaltrials_2021_trec-ct-2021", "qrels", trust_remote_code=True)

# Build a dictionary mapping query_id to query text.
topics = {record["query_id"]: record["text"] for record in queries_dataset}

# Build a dictionary mapping query_id to a list of (doc_id, relevance) tuples.
qrels_dict = {}
for record in qrels_dataset:
    qid = record["query_id"]
    doc_id = record["doc_id"]
    relevance = record["relevance"]
    qrels_dict.setdefault(qid, []).append((doc_id, relevance))

# List to store the total number of relevant docs (relevance > 0) per query.
relevant_counts = []

print("=== Relevance Counts for Each Query ===\n")
for qid in sorted(topics.keys()):
    # Initialize counts for each relevance level.
    counts = {0: 0, 1: 0, 2: 0}
    for doc_id, rel in qrels_dict.get(qid, []):
        # Convert the relevance to an integer.
        rel_val = int(rel)
        counts[rel_val] += 1
    total_relevant = counts[1] + counts[2]
    relevant_counts.append(total_relevant)
    print(f"Query {qid}:")
    print(f"  Not Relevant (0): {counts[0]}")
    print(f"  Possibly Relevant (1): {counts[1]}")
    print(f"  Definitely Relevant (2): {counts[2]}")
    print(f"  Total Relevant (relevance > 0): {total_relevant}")
    print("-" * 50)

# Compute overall statistics for relevant documents per query.
total_queries = len(topics)
mean_relevant = statistics.mean(relevant_counts) if relevant_counts else 0
median_relevant = statistics.median(relevant_counts) if relevant_counts else 0
std_relevant = statistics.stdev(relevant_counts) if len(relevant_counts) > 1 else 0

print("\n=== Overall Relevance Statistics ===\n")
print(f"Total number of queries: {total_queries}")
print(f"Mean number of relevant docs per query: {mean_relevant:.2f}")
print(f"Median number of relevant docs per query: {median_relevant}")
print(f"Standard deviation: {std_relevant:.2f}")

# -----------------------------------------------
# Compute statistics on query lengths (in words)
# -----------------------------------------------
# Calculate the number of words for each query.
query_lengths = [len(text.split()) for text in topics.values()]

if query_lengths:
    mean_query_len = statistics.mean(query_lengths)
    median_query_len = statistics.median(query_lengths)
    std_query_len = statistics.stdev(query_lengths) if len(query_lengths) > 1 else 0
    min_query_len = min(query_lengths)
    max_query_len = max(query_lengths)
else:
    mean_query_len = median_query_len = std_query_len = min_query_len = max_query_len = 0

print("\n=== Query Length Statistics (in words) ===\n")
print(f"Mean query length: {mean_query_len:.2f}")
print(f"Median query length: {median_query_len}")
print(f"Standard deviation: {std_query_len:.2f}")
print(f"Minimum query length: {min_query_len}")
print(f"Maximum query length: {max_query_len}")