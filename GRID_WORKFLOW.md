# Running on the CLSP Grid

This project was originally run on the JHU CLSP SLURM grid under one account. The submission scripts are now versioned in [`grid/`](grid/) in this repo — they used to live only in the original author's home directory (`~`), outside git entirely, which meant no one else could see or run them. If you find older copies of these scripts elsewhere on the grid outside this repo, treat `grid/` here as the source of truth going forward.

## Background: what these scripts are

Each script is a SLURM batch script (`#SBATCH` directives at the top configure the job — partition, GPU, memory, time limit), submitted with:

```bash
sbatch grid/<script>.sh
```

Check job status with `squeue -u $USER`; output/error logs land in the directory you submitted from, named `{job-name}_{job-id}.out`/`.err` per the `#SBATCH --output`/`--error` directives.

## Before running anything: edit these

**Every script in `grid/` has a `CONFIGURATION` block near the top** with the account-specific values you need to change — at minimum `PROJECT_DIR` (your clone of this repo) and `CONDA_ENV`. Read each script's block before submitting; don't assume the defaults match your setup.

Beyond the scripts themselves, some of the **Python scripts they invoke** also have hardcoded, account-specific paths that are not exposed as shell variables (they're constants inside the `.py` file). These are inventoried with exact file:line references in [USAGE.md](USAGE.md)'s Configuration Reference — check that list before your first run, since a script can complete "successfully" while writing results somewhere you don't expect (or reading from a keyword cache / corpus that doesn't exist under your account).

## Scripts and what they map to

| Script | Pipeline stage (see [PIPELINE.md](PIPELINE.md)) | Partition | Typical runtime |
|---|---|---|---|
| `grid/run_create_filtered_corpus.sh` | Stage 0: corpus prep (doctor-reviews only, one-time) | cpu | ~few min |
| `grid/run_generate_keywords.sh` | Stage 3: keyword generation | gpu | ~fast |
| `grid/run_search.sh` | Stage 4: main pipeline (`end_to_end_evaluation.py`) | gpu | hours (see USAGE.md Expected Runtime) |
| `grid/run_hicode_eval.sh` | Stage 4b: HiCode evaluation | gpu | up to ~12h for 11 queries |
| `grid/run_regenerate_diversity_plots.sh` | Legacy one-off utility, not a core stage | cpu | <1 min (if it still works — see the script's header comment) |

There is no grid script for Stage 1 (indexing) or Stage 2 (search/retrieval evaluation, `search.py`/`query_expansion.py` standalone tools) — those either run transparently as part of Stage 4, or are research tools meant to be run interactively/locally rather than batch-submitted.

## Selecting dataset / topic model on the grid

`grid/run_search.sh` runs `end_to_end_evaluation.py`, which reads `DATASET` and `MODEL` from the environment (falling back to `trec-covid`/`bertopic`). To run a different combination without editing the script, pass them through at submission time:

```bash
sbatch --export=ALL,DATASET=doctor-reviews,MODEL=lda grid/run_search.sh
```

Valid values: `DATASET` ∈ {`trec-covid`, `doctor-reviews`}, `MODEL` ∈ {`bertopic`, `lda`, `topicgpt`}. (HiCode is not selected via `MODEL` — it's a separate script, `grid/run_hicode_eval.sh`.) For `topicgpt`, also export `OPENAI_API_KEY`.

## External dependencies you can't fix by editing a path

- **HiCode's own output** is produced by a separate process outside this repo, and the existing results were generated against `/export/fs06/mzhong8/hicode/results/assignment` — a directory owned by another lab member on the shared grid filesystem. If you don't have read access, you'll need either access granted, or to run HiCode yourself and point `HICODE_EXTERNAL_DIR` in `grid/run_hicode_eval.sh` at your own output.
- **`acquire-gpu`** (`/home/gqin2/scripts/acquire-gpu`, referenced in `run_search.sh`/`run_hicode_eval.sh`/`run_generate_keywords.sh`) is a shared, grid-wide utility for avoiding GPU allocation race conditions — not account-specific, safe to leave as-is if it exists on the grid you're using. If it doesn't exist on your grid, the scripts fall back to plain SLURM GPU allocation and print a warning.
- **`OPENAI_API_KEY`** — required only for `TOPIC_MODEL_TYPE = "topicgpt"`. Export it in your shell before `sbatch`, or add `--export=ALL,OPENAI_API_KEY=...` — never commit a key into a script.

## Known naming quirk (documented, not a bug)

`run_search.sh`'s SLURM job name and log-file prefix are `search_evaluation`, a holdover from when this script (or an earlier version of it) may have targeted `src/search.py`'s standalone retrieval evaluator. It currently runs the full `end_to_end_evaluation.py` pipeline. The job name is cosmetic — it doesn't affect what the script does — but don't be misled by `search_evaluation_*.out` log filenames into thinking they're `search.py` output.
