import os
from datasets import load_dataset
from bertopic import BERTopic
from tqdm import tqdm

def load_trec_covid_corpus():
    """
    Load the TREC-COVID corpus using the Hugging Face datasets library.
    Returns the corpus dataset.
    """
    print("Loading datasets...")
    corpus_dataset = load_dataset("BeIR/trec-covid", "corpus")["corpus"]
    print("Datasets loaded.")
    return corpus_dataset

def preprocess_corpus(corpus_dataset, max_docs=None):
    """
    Preprocess the corpus by combining the title and text of each document.
    If max_docs is specified, only process up to max_docs documents.
    Returns a list of combined document texts.
    """
    corpus_texts = []
    for i, doc in enumerate(tqdm(corpus_dataset, desc="Preprocessing corpus")):
        if max_docs is not None and i >= max_docs:
            break
        title = doc.get("title", "")
        text = doc.get("text", "")
        combined = f"{title}\n\n{text}"
        corpus_texts.append(combined)
    return corpus_texts

def run_bertopic(corpus_texts):
    """
    Fit the BERTopic model on the provided corpus texts.
    Returns the fitted BERTopic model and the list of topics.
    """
    print("Fitting BERTopic model...")
    topic_model = BERTopic()
    topics, probs = topic_model.fit_transform(corpus_texts)
    print("BERTopic model fitted.")
    return topic_model, topics

def main():
    # Set the maximum number of documents to process. Set to None to process all documents.
    max_docs = 1000  # Modify this value as needed

    corpus_dataset = load_trec_covid_corpus()
    corpus_texts = preprocess_corpus(corpus_dataset, max_docs=max_docs)
    topic_model, topics = run_bertopic(corpus_texts)

    # Example: print the top 5 topics
    print("\nTop 5 topics:")
    for topic_id in topic_model.get_topic_freq().head(5)['Topic']:
        print(f"Topic {topic_id}: {topic_model.get_topic(topic_id)}")

if __name__ == "__main__":
    main()