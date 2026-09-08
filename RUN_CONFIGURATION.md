# Run Configuration Reference

This is the verified record of what configuration has actually been used to produce the current results, and the checklist for standing up a new dataset against the same methodology.

**This is not the same document as `USAGE.md`.** `USAGE.md` documents the knobs that exist and how to turn them. This document records the specific *values* that produced `/home/srangre1/results/`, verified against source code and cached output — and is explicit about the one place those two didn't match.

## Read this first: don't trust `config.json`

`config.json` gets overwritten on every run, including a metrics-only rerun of an old cached model — so the copy sitting on disk next to an LDA or BERTopic result often reflects a *later, unrelated* run's settings, not the ones that actually produced it. Every `config.json` on disk right now shows TopicGPT's params regardless of the model type, which is the tell that this has happened. Don't use it to answer "what did this run actually use."

The values in this document were recovered from the cached topic-model output and source defaults instead — see each section for how.

---

## Fixed, verified settings (pipeline-wide, both datasets)

| Setting | Value | Verified via |
|---|---|---|
| Sample size | 1000 docs/method | `config.json` (consistent across all configs checked) |
| Random seed | 42 | `config.json`, `TopicModelWrapper` LDA default |
| Pipeline embedding model (`EMBEDDING_MODEL`) | `all-mpnet-base-v2` | `config.json`, drives retrieval/BERTopic/KeyBERT |
| Metrics embedding models | Both `all-mpnet-base-v2` and `BAAI/bge-base-en-v1.5` computed | 168 tagged result subdirectories on disk |
| Cross-encoder model configured | `cross-encoder/ms-marco-MiniLM-L-6-v2` | `config.json` |
| Cross-encoder actually used in Stage 4 sampling | **Never** — see Caveats | `grep use_cross_encoder` → `False` at all 7 call sites in `end_to_end_evaluation.py` |
| Force flags (observed) | `samples=False, topics=False, evaluation=True` | Consistent across every `config.json` checked — i.e. official runs typically re-evaluate against already-cached samples/topics, not full cold runs |

---

## Stage 1 — Indexing (verified defaults, not overridden by the main pipeline)

`end_to_end_evaluation.py`'s `_initialize_search_indices()` calls `IndexManager.build_bm25_index()`/`.build_sbert_index()` passing only `dataset_name`, `model_name=self.embedding_model_name`, `batch_size=64`, and `force_reindex=False` — everything else falls through to `IndexManager`'s own signature defaults in `src/search.py`:

| Parameter | Value | Source |
|---|---|---|
| BM25 `b_param` | 0.75 | `IndexManager.build_bm25_index` default |
| BM25 `k1_param` | 1.5 | same |
| BM25 `epsilon` | 0.25 | same |
| BM25 `stemmer` | `porter` | same |
| SBERT `batch_size` | 64 | explicit in `_initialize_search_indices` |
| SBERT `max_seq_length` | model default (not set) | `build_sbert_index` default `None` |
| SBERT `normalize_embeddings` | `True` | `build_sbert_index` default |
| `force_reindex` | `False`, hardcoded — no config flag in the main pipeline to force a rebuild | `_initialize_search_indices` |

**Note:** `build_sbert_index`'s own function-signature default for `model_name` is `sentence-transformers/msmarco-MiniLM-L6-cos-v5` — this is never actually used, since the main pipeline always passes `model_name=self.embedding_model_name` explicitly. Don't be misled by that default if reading the function signature directly.

---

## Stage 3 — Keyword generation (verified via cache filename + source)

Identical parameters used by both `keyword_extraction.py main()` (TREC-COVID) and `generate_doctor_review_keywords.py` (doctor-reviews) — confirmed because the cache filename is a deterministic function of these exact values (`KeywordExtractor._generate_keywords_cache_path()`):

| Parameter | Value |
|---|---|
| `num_keywords` | 10 |
| `diversity` | 0.7 |
| `top_k_docs` (retrieval pool for extraction) | 1000 |
| `top_n_docs_for_extraction` | 10 |
| `keyphrase_ngram_range` | (1, 2) |
| Embedding model | `all-mpnet-base-v2` |
| Retrieval method for candidate docs | Hybrid, Simple-Sum fusion |

**Coverage:** TREC-COVID — all 50 possible query IDs cached (only 15 are used by `DATASET_CONFIGS`). Doctor-reviews — only queries `1`-`6` of 11 cached; queries `7`-`11` fall back to `direct_retrieval` at runtime (see `CACHING.md`). **If standing up a new dataset, generate keywords for every query ID you intend to use before running Stage 4** — this gap is the single most common way a new dataset silently ends up with a degenerate `query_expansion` method.

---

## Stage 4 — Per-sampling-method parameters (verified, file:line in `src/end_to_end_evaluation.py`)

| Method | Verified parameters |
|---|---|
| `random_uniform` (`:1416`) | No parameters |
| `keyword_search` (`:1550`) | BM25 only, `top_k=SAMPLE_SIZE` |
| `sbert` (`:1580`) | SBERT only |
| `direct_retrieval` (`:1447`) | Hybrid, `HybridStrategy.SIMPLE_SUM`, `use_mmr=False`, `use_cross_encoder=False` |
| `direct_retrieval_mmr` (`:1610`) | Candidate pool = 5000 (hardcoded), `lambda_param=0.3` (hardcoded, 30% relevance / 70% diversity) |
| `query_expansion` (`:1478`) | Weighted RRF: 0.7 baseline + 0.3 split across keywords (hardcoded); `HybridStrategy.SIMPLE_SUM` for baseline and each keyword search |
| `retrieval_random` (`:1700`) | `pool_size=5000` (function default, never overridden — confirmed the one call site at `:3826` passes no argument) |

None of these per-method internals are exposed as `main()`-level config — changing any of them means editing the method body directly.

---

## Stage 4 — Per-topic-model parameters (verified against `topic_models.py`, not `config.json`)

### BERTopic

Confirmed via `TopicModelWrapper._fit_bertopic()` — `main()`'s `TOPIC_MODEL_PARAMS` does not currently override any of these (commented out in source):

| Parameter | Value |
|---|---|
| `min_cluster_size` (HDBSCAN) | 5 |
| `metric` (HDBSCAN) | euclidean |
| `cluster_selection_method` (HDBSCAN) | eom |
| `prediction_data` | True |
| Dimensionality reduction | BERTopic's library-default UMAP (not overridden anywhere in this codebase) |
| Vectorizer | `CountVectorizer(stop_words='english', min_df=2, ngram_range=(1,2), max_features=10000)` |
| `calculate_probabilities` | True |

### LDA

`TopicModelWrapper._fit_lda()`'s own fallback defaults (used whenever a key isn't in `topic_model_params`):

| Parameter | Fallback value |
|---|---|
| `n_topics` | 20 (or resolved via `_get_bertopic_n_topics()` if `topic_model_params["n_topics"] == "auto"` is set at fit time, matching BERTopic's discovered count for that method) |
| `alpha` | symmetric |
| `eta` | 0.01 |
| `passes` | 15 |
| `iterations` | 100 |
| `random_state` | 42 |
| `workers` | 20 |

`_get_bertopic_n_topics()` previously had its dataset hardcoded to `trec-covid` regardless of which dataset was actually running, so `"auto"` could resolve against the wrong dataset's BERTopic results on doctor-reviews. This has been fixed (the path is now derived from the current run's own directory) and the fix is pushed. Existing cached doctor-reviews LDA topic models predate the fix; refit them (`FORCE_REGENERATE_TOPICS = True` with `"n_topics": "auto"` set) if you need current, correctly-matched counts.

### TopicGPT

Currently the active block in `main()`'s `TOPIC_MODEL_PARAMS` (confirmed matching `config.json` for topicgpt runs):

| Parameter | Value |
|---|---|
| `generation_model` | `gpt-4o-mini` |
| `assignment_model` | `gpt-4o-mini` |
| `generation_sample_size` | 500 |
| `verbose` | True |

### Vocabulary parameters shared across all three word/BoW-vocabulary-based models

`min_df=2`, `ngram_range=(1,2)`, `max_features=10000` — confirmed identical in BERTopic's `CountVectorizer`, LDA's dictionary filtering, and TopicGPT's vocabulary step.

### HiCode

Not fit by this repo — evaluates an external run. Conversion settings (from its own `config.json`, which is a distinct, reliable schema — not the shared `TOPIC_MODEL_PARAMS` problem above): `multi_label_strategy: take_first_topic`, `stopword_filtering: lexical_metrics_only`.

---

## Statistical testing (verified defaults)

| Parameter | Value | Source |
|---|---|---|
| BH-FDR alpha | 0.05 | Default across `benjamini_hochberg_correction`, `run_pairwise_statistical_tests`, `run_pairwise_coverage_statistical_tests` — not overridden anywhere |
| Coverage significance thresholds | 0.5, 0.6, 0.7 | `run_pairwise_coverage_statistical_tests` |
| Metrics tested in aggregate | 11 (see `METRICS_GUIDE.md`) | `run_aggregate_statistical_analysis`'s default `metrics_to_test` |

---

## Dataset-specific configuration

| | TREC-COVID | Doctor-Reviews |
|---|---|---|
| Query IDs used | `2,9,10,13,18,21,23,24,26,27,34,43,45,47,48` (15) | `1`-`11` (11) |
| Keyword cache path | `/home/srangre1/cache/keywords/keybert_k10_div0.7_top10docs_mpnet_k1000_ngram1-2.json` | `/home/srangre1/cache/keywords/doctor_reviews_keybert.json` (only covers queries 1-6) |
| Corpus source | HuggingFace `BeIR/trec-covid`, loaded directly | Filtered via `create_filtered_doctor_corpus.py` from raw source at `/export/fs06/mzhong8/doctor_review_{metadata,corpus}` |
| QRELs (relevance judgments) | Yes — `BeIR/trec-covid-qrels` | **No** — `end_to_end_evaluation.py:1377-1378` has explicit handling for `qrels_dataset is None`; `relevant_concentration` reads 0 for this dataset |

---

## New dataset checklist

1. **Decide whether a corpus-prep step is needed.** If the raw data needs filtering/reshaping before use (like doctor-reviews), write an equivalent to `create_filtered_doctor_corpus.py`; if it's already in a loadable HF-style format (like TREC-COVID), skip this.
2. **Add a loader function** to `dataset_loaders.py`'s `load_dataset()` if/else ladder.
3. **Add an entry to `DATASET_CONFIGS`** in `end_to_end_evaluation.py`'s `main()`: `query_ids` (decide the full list up front) and `keyword_cache_path`.
4. **Generate the keyword cache for every query ID in that list** (Stage 3) before running Stage 4 — not a subset. Verify by checking the cache JSON's key count matches `len(query_ids)`, not by assuming the script covered everything.
5. **Determine QRELs availability.** If none exist, `relevant_concentration` will read 0 for every method — decide whether that's acceptable or whether a proxy relevance signal is needed.
6. **Run BERTopic first**, for every method, before touching LDA — LDA's `n_topics="auto"` path depends on `per_method_summary.csv` already existing for BERTopic. Explicitly set `"n_topics": "auto"` in `TOPIC_MODEL_PARAMS` before running LDA (it is not currently the default active block).
7. **HiCode is a separate decision.** It requires an external HiCode run (owned by whoever has access to run it) before `evaluate_hicode.py` has anything to evaluate.
8. **Everything in "Fixed, verified settings" above should stay unchanged** — sample size, embedding models, cross-encoder model (even though unused), vocabulary parameters, statistical test settings. Changing any of these breaks comparability with the existing TREC-COVID/doctor-reviews results.

---

## Caveats and open issues

- **Cross-encoder model is configured but never invoked** in Stage 4 sampling (`use_cross_encoder=False` at all 7 call sites). It's used in Stage 2's standalone `search.py` research tool, not in the pipeline that produced the current results.
- **`config.json` is not a reliable per-run record** — see the top of this document. If this is a recurring pain point, consider having `save_config()` record the actually-*resolved* topic model parameters (post `"auto"` resolution) rather than the raw input dict, and write it once at fit time rather than on every re-evaluation-only run.

---

## Related documentation

- [USAGE.md](USAGE.md) — what each config knob is and how to change it
- [CACHING.md](CACHING.md) — cache tiers and invalidation behavior referenced throughout
- [PIPELINE.md](PIPELINE.md) — architecture and stage dependencies
- [METRICS_GUIDE.md](METRICS_GUIDE.md) — the 11 metrics covered by aggregate statistical testing
