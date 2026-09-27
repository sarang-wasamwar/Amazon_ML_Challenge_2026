# 📊 Amazon ML Challenge 2026 — Project Status Report

> **Generated:** 27 September 2026 | **Workspace:** `Amazon_ML_Challenge_2026`
> **Objective:** Business Entity Resolution optimized for **Macro F₀.₅**

---

## ✅ What Has Been Built

All five pipeline modules are fully implemented, tested, and runnable. Below is the complete inventory of delivered work.

---

## 📁 Repository Structure

```
Amazon_ML_Challenge_2026/
├── student_resource/
│   ├── code/
│   │   └── business_entity_resolution/
│   │       ├── src/
│   │       │   ├── preprocess.py      ✅  Text normalization & address standardization
│   │       │   ├── blocking.py        ✅  Country-partitioned TF-IDF candidate generation
│   │       │   ├── features.py        ✅  14-feature pairwise similarity extraction
│   │       │   ├── model.py           ✅  LightGBM + F₀.₅ threshold tuner + GPU support
│   │       │   └── main.py            ✅  End-to-end pipeline orchestrator
│   │       ├── requirements.txt       ✅  Pinned production dependencies
│   │       └── README.md              ✅  Execution guide
│   ├── dataset/
│   │   ├── train/   (train_source1/2/3.tsv, train_ground_truth.tsv)
│   │   └── test/    (test_source1/2/3.tsv)
│   ├── output/
│   │   ├── matching_results.tsv       ⚠️  Exists but from partial/sampled run
│   │   └── candidate_pairs.tsv        ⚠️  Exists but from partial/sampled run
│   └── utils/
│       └── validate_submission.py     (Official validator — provided by organizers)
├── README.md                          ✅  Full project documentation
└── .gitignore                         ✅  Excludes datasets & large outputs
```

---

## 🗂️ Dataset Overview

| File | Records | Size |
|---|---|---|
| `train_source1.tsv` | ~1M+ | Large |
| `train_source2.tsv` | ~4M+ | Large |
| `train_source3.tsv` | ~5M+ | Large |
| `train_ground_truth.tsv` | ~1M+ | Large |
| `test_source1.tsv` | **1,732,544** | ~25 MB |
| `test_source2.tsv` | **4,887,273** | ~500 MB |
| `test_source3.tsv` | **5,082,316** | ~500 MB |

**Countries covered:** United States (US), India (IN), France (FR)

---

## 🧱 Module-by-Module Breakdown

### 1. `preprocess.py` — Text Normalization

**Status:** ✅ Complete

- Loads TSVs with `sep='\t'`, `dtype=str`, `.fillna("")`
- `clean_business_name()`: Lowercase → punctuation removal → **legal suffix stripping**
  - Strips: `corp`, `inc`, `ltd`, `pvt`, `llc`, `pc`, `co`, `corporation`, `incorporated`, `limited`, `private`, `company`
- `standardize_address()`: Multi-locale address normalization
  - **US:** `rd→road`, `st→street`, `ave→avenue`, `blvd→boulevard`, `ste→suite`, directionals (N/S/E/W)
  - **India:** `nr→near`, `opp→opposite`, `b/h→behind`, city aliases (`bombay→mumbai`, `calcutta→kolkata`, `madras→chennai`, `bangalore→bengaluru`)
  - **France:** `r→rue`, `bd→boulevard`, `bat→batiment`, `res→residence`
- `preprocess_dataframe()`: Combines into `clean_name`, `clean_address`, `combined_text` columns
- Handles missing columns gracefully

---

### 2. `blocking.py` — Candidate Generation

**Status:** ✅ Complete | Memory-efficient

**Algorithm:** Country-partitioned TF-IDF character n-gram cosine similarity

**Key design decisions made:**

| Decision | Rationale |
|---|---|
| Character 3-4 grams (`char_wb`) | Robust to typos, abbreviations, suffix variants |
| Country partitioning | Reduces S1 vs S23 comparison space from ~10M to per-country pools |
| Sub-batch size = 500 | Bounds RAM per iteration, avoids the 307 GiB OOM that occurred with full matrix dot |
| `min_similarity = 0.25` | Prunes zero/noise entries before building pair list |
| `top_n = 15` (default) | Retrieves top-15 candidates per S1 entity |
| `float32` sparse matrices | 2× memory reduction vs float64 |
| Fallback to full pool | If no S23 records match the country, compare against all S23 |

**Fixed bugs (from earlier sessions):**
- ~~`X_batch.dot(X_s23.T)` OOM~~ — replaced with 500-record sub-batches
- ~~Windows multi-process re-spawning duplicate prints~~ — removed multiprocessing guard

---

### 3. `features.py` — Feature Engineering

**Status:** ✅ Complete | 14 features per candidate pair

| # | Feature | Method |
|---|---|---|
| 1 | `name_levenshtein` | Levenshtein normalized similarity on `clean_name` |
| 2 | `name_token_sort` | RapidFuzz `token_sort_ratio` / 100 |
| 3 | `name_token_set` | RapidFuzz `token_set_ratio` / 100 |
| 4 | `address_levenshtein` | Levenshtein normalized similarity on `clean_address` |
| 5 | `address_token_sort` | RapidFuzz `token_sort_ratio` / 100 |
| 6 | `address_token_set` | RapidFuzz `token_set_ratio` / 100 |
| 7 | `address_jaccard` | Token Jaccard similarity on `clean_address` |
| 8 | `combined_token_set` | RapidFuzz `token_set_ratio` on `combined_text` |
| 9 | `tfidf_cosine_sim` | Cosine similarity score passed down from blocking |
| 10 | `digit_match_count` | Count of shared digit strings between records |
| 11 | `digit_jaccard` | Jaccard similarity of extracted digit sets |
| 12 | `exact_name_match` | Binary: 1.0 if `clean_name` strings identical |
| 13 | `exact_address_match` | Binary: 1.0 if `clean_address` strings identical |
| 14 | `country_match` | Binary: 1.0 if countries match |

- Uses dict-based O(1) entity lookup (indexed by `entity_id`)
- Pure Python loop for max flexibility (no vectorized assumption issues)

---

### 4. `model.py` — LightGBM + F₀.₅ Threshold Tuner

**Status:** ✅ Complete | GPU-ready with CPU fallback

#### Training

- **Model:** `lgb.LGBMClassifier`
- **Imbalance handling:** `is_unbalance=True` (auto-adjusts class weights)
- **Early stopping:** `stopping_rounds=30` on validation loss
- **Hyperparameters:**
  ```
  n_estimators    = 400
  learning_rate   = 0.05
  num_leaves      = 31
  max_depth       = 6
  subsample       = 0.8
  colsample_bytree= 0.8
  ```
- **Train/Val split:** 80/20 entity-level split (entity IDs don't bleed across sets)

#### GPU Acceleration

```python
device="gpu", gpu_platform_id=0, gpu_device_id=0
```
- ✅ Attempts GPU training first
- ✅ Falls back to CPU gracefully if GPU driver unavailable
- ✅ Prints confirmation when GPU mode is active

#### F₀.₅ Threshold Tuning

- Grid search over `τ ∈ [0.60, 0.85]` with 0.01 step
- Evaluates **Macro F₀.₅** on the entity-level validation split
- `compute_macro_f05()` correctly handles:
  - Singleton entities (true_set empty ∧ pred_set empty → F₀.₅ = 1.0)
  - False positive singletons (true_set empty ∧ pred_set non-empty → F₀.₅ = 0.0)

#### Inference

- `predict_and_save_matching_results()` applies tuned `τ` to test probabilities
- Guarantees **all 1,732,544 test S1 entity IDs** appear in output (singletons → empty string)

---

### 5. `main.py` — Pipeline Orchestrator

**Status:** ✅ Complete | 6-step pipeline

```
[STEP 1/6]  Load datasets (positive-aware sampling if --sample-size given)
[STEP 2/6]  Preprocess S1, S2, S3 for train and test
[STEP 3/6]  Blocking / candidate generation on TRAINING set
[STEP 4/6]  Feature extraction + LightGBM training + threshold tuning
[STEP 5/6]  Blocking + feature extraction + inference on TEST set
[STEP 6/6]  Run official submission validator subprocess
```

**CLI arguments:**

| Argument | Default | Description |
|---|---|---|
| `--train-dir` | `student_resource/dataset/train` | Training data path |
| `--test-dir` | `student_resource/dataset/test` | Test data path |
| `--output-dir` | `student_resource/output` | Output path |
| `--top-n` | `15` | Top-N candidates per S1 entity |
| `--sample-size` | `None` | Fast dev run (e.g. `5000`) |
| `--skip-validation` | `False` | Skip validator subprocess |

**Positive-aware sampling (when `--sample-size` is set):**
- Samples 70% positives + 30% singletons from `train_ground_truth.tsv`
- Guarantees label=1 pairs exist in training data
- Pads final output to include all 1.73M test S1 IDs

---

### 6. `requirements.txt` — Pinned Dependencies

**Status:** ✅ Installed in Python 3.13 environment

```
lightgbm==4.7.0
scikit-learn==1.9.1
rapidfuzz==3.14.6
pandas==3.0.6
numpy==2.5.3
scipy==1.18.1
```

---

## 📈 Current Output Status

| File | Rows | Status |
|---|---|---|
| `matching_results.tsv` | **1,732,544** rows | ⚠️ From partial run — only **205 non-empty** matches |
| `candidate_pairs.tsv` | **1,732,544** rows | ⚠️ Only **4,717** S1 entities have candidates |
| `test_cand.tsv` | 2 rows | ✅ Test artifact from blocking.py `__main__` |

> **Root cause of low matches:** The pipeline was last run with a limited/sampled test execution. A full `python main.py` (no `--sample-size`) is needed to process all 1.73M test S1 entities against the ~10M S2+S3 test pool.

---

## 🐛 Bugs Fixed During Development

| Bug | Fix Applied |
|---|---|
| `ModuleNotFoundError: pandas` | Installed all dependencies via `pip install` |
| `ArrayMemoryError: 307 GiB` in blocking | Replaced full matrix dot with 500-record sub-batches |
| Windows process spawning duplicate prints | Removed multiprocessing; single-process chunked approach |
| `--sample-size` had no positives (label=1) | Positive-aware sampling: 70% from positive GT rows |
| LightGBM `Contains only one class` warning | Validate class count before fit; graceful fallback |
| LightGBM verbose output flooding | Set `verbose=-1` globally |
| GPU crash on unsupported driver | `try/except` GPU → CPU fallback with warning message |

---

## 🚀 How to Run

### Quick Validation Run (5,000 records)
```bash
cd c:\Users\wasam\__Coding_with_SaranG__\Amazon_ML_Challenge_2026
py -3.13 student_resource/code/business_entity_resolution/src/main.py --sample-size 5000
```

### Full Pipeline Run
```bash
py -3.13 student_resource/code/business_entity_resolution/src/main.py
```

### Validate Submission Output
```bash
py -3.13 student_resource/utils/validate_submission.py \
  --matching student_resource/output/matching_results.tsv \
  --candidate student_resource/output/candidate_pairs.tsv \
  --test-dir student_resource/dataset/test
```

---

## ⚠️ Remaining Work / Open Items

| Priority | Item | Notes |
|---|---|---|
| 🔴 HIGH | **Run full pipeline** | `main.py` with no `--sample-size` to generate complete submission |
| 🔴 HIGH | **Validate submission** | Run official validator on final output files |
| 🟡 MED | **F₀.₅ score verification** | Check validation macro F₀.₅ from training logs |
| 🟡 MED | **Package submission zip** | `Compress-Archive output, code, Documentation_template.md` |
| 🟢 LOW | **Feature enhancement** | Phonetic matching (Soundex/Metaphone), postal code extraction |
| 🟢 LOW | **FAISS ANN blocking** | Replace TF-IDF dot product with approximate nearest neighbors for speed |
| 🟢 LOW | **Cross-country fallback candidates** | Currently singletons if country not in S2/S3 (uses full pool) |

---

## 🎯 Optimization Target

```
F₀.₅ = (1.25 × Precision × Recall) / (0.25 × Precision + Recall)
```

F₀.₅ weights **precision 2× more than recall** — so it is better to predict fewer high-confidence matches than many noisy ones. The threshold tuner (`τ ∈ [0.60, 0.85]`) directly optimizes this on the validation split.

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.13 |
| ML Model | LightGBM 4.7.0 |
| Text Vectorization | TF-IDF (scikit-learn) |
| String Matching | RapidFuzz 3.14.6 |
| Edit Distance | Levenshtein (rapidfuzz.distance) |
| Data Processing | Pandas 3.0.6 / NumPy 2.5.3 |
| Sparse Operations | SciPy 1.18.1 CSR matrices |
| Evaluation | Macro F₀.₅ (custom) |
| GPU Acceleration | LightGBM CUDA (`device='gpu'`) |
| Data Format | TSV (tab-separated) |
| Version Control | Git / GitHub |
