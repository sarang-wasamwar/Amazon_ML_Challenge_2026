# Business Entity Resolution — Amazon ML Challenge 2026

An end-to-end Machine Learning pipeline for large-scale Business Entity Resolution across multiple independent data sources (Source 1, Source 2, and Source 3). Designed to optimize the **Macro-Averaged F_0.5 Score** (favoring precision 2x over recall) while scaling efficiently across millions of records.

---

## 📁 Repository Structure

```text
Amazon_ML_Challenge_2026/
├── student_resource/
│   ├── code/
│   │   └── business_entity_resolution/
│   │       ├── src/
│   │       │   ├── preprocess.py     # String normalization & legal suffix stripping
│   │       │   ├── blocking.py       # Country-partitioned TF-IDF n-gram candidate generation
│   │       │   ├── features.py       # Pairwise string & address similarity extraction
│   │       │   ├── model.py          # LightGBM training & Macro F_0.5 threshold tuner
│   │       │   └── main.py           # End-to-end pipeline orchestrator & test inference
│   │       ├── requirements.txt      # Pinned Python dependencies
│   │       └── README.md             # Project execution guide
│   ├── dataset/                      # Ignored in git (stored locally)
│   │   ├── train/
│   │   └── test/
│   ├── output/                       # Generated predictions (ignored in git)
│   │   ├── matching_results.tsv
│   │   └── candidate_pairs.tsv
│   ├── utils/
│   │   └── validate_submission.py    # Official format & structure validator
│   └── Documentation_template.md     # Methodology write-up for submission
├── .gitignore                        # Git exclusion rules for large datasets/outputs
└── README.md
```

---

## ⚙️ Methodology & Architecture

### 1. Preprocessing & Normalization

**`preprocess.py`**

- Strips common legal suffixes such as:
  - `corp`
  - `inc`
  - `ltd`
  - `pvt`
  - `llc`
  - `corporation`
  - `limited`
- Normalizes whitespace.
- Converts text to lowercase.
- Standardizes common street and address tokens.
- Dynamically extracts country labels across the US, India, and France.

### 2. Blocking / Candidate Generation

**`blocking.py`**

- Uses character n-gram **TF-IDF (3–5 grams)** for candidate generation.
- Partitions records by `country` to significantly reduce the search space.
- Uses memory-efficient, chunked sparse matrix multiplication.
- Retrieves the top candidate matches for each Source 1 entity.
- Avoids constructing the full pairwise similarity matrix in memory.

### 3. Feature Engineering

**`features.py`**

The candidate pairs are represented using multiple complementary similarity features:

- Character-level **Levenshtein similarity ratio**
- RapidFuzz `token_sort_ratio`
- RapidFuzz `token_set_ratio`
- Token-based **Jaccard similarity**
- TF-IDF cosine similarity
- Street-token overlap
- Postal-code digit overlap
- Address similarity features

These features combine exact, token-level, and character-level matching signals.

### 4. Model Training & F_0.5 Threshold Tuning

**`model.py`**

- Uses a **LightGBM Binary Classifier**.
- Handles the highly imbalanced candidate-pair distribution using `is_unbalance=True`.
- Creates a validation split for threshold selection.
- Searches for an optimal decision threshold `τ` within:

```text
[0.60, 0.85]
```

- Selects the threshold based specifically on the **Macro-Averaged F_0.5 score**.
- F_0.5 gives twice as much importance to precision as recall.
- Entities without a candidate exceeding the selected threshold receive an empty prediction list.

---

## 🚀 Quick Start Guide

### 1. Environment Setup

Ensure you are running **Python 3.10+**.

Install the required dependencies:

```bash
pip install -r student_resource/code/business_entity_resolution/requirements.txt
```

### 2. Run Pipeline — Sample Test

Run a fast dry-run using a sample subset, for example **5,000 entities**, to verify the pipeline and memory usage:

```bash
python student_resource/code/business_entity_resolution/src/main.py --sample-size 5000
```

This is recommended before running the complete dataset.

### 3. Run Pipeline — Full Dataset

Execute the complete end-to-end pipeline:

```bash
python student_resource/code/business_entity_resolution/src/main.py
```

The pipeline generates:

```text
student_resource/output/matching_results.tsv
student_resource/output/candidate_pairs.tsv
```

---

## 🔍 Submission Validation

Before submitting, validate the generated prediction files:

```bash
python student_resource/utils/validate_submission.py \
  --matching student_resource/output/matching_results.tsv \
  --candidate student_resource/output/candidate_pairs.tsv \
  --test-dir student_resource/dataset/test
```

Expected result:

```text
PASS — no blocking issues found. Safe to submit.
```

---

## 📦 Packaging for Final Submission

Create the final submission archive from the workspace root.

### PowerShell

```powershell
cd student_resource

Compress-Archive `
  -Path output, code, Documentation_template.md `
  -DestinationPath ..\team_submission.zip `
  -Force

cd ..
```

This generates:

```text
team_submission.zip
```

in the project root.

---

## 🎯 Optimization Objective

The pipeline is designed around the competition's **Macro-Averaged F_0.5 Score**.

Unlike F_1, F_0.5 places greater emphasis on precision:

```text
F₀.₅ = (1 + 0.5²) × (Precision × Recall)
       ──────────────────────────────────
       (0.5² × Precision) + Recall
```

This makes conservative, high-confidence entity matches particularly important.

---

## 🧠 End-to-End Pipeline

```text
                ┌─────────────────────┐
                │     Raw Datasets    │
                │ Source 1 / 2 / 3    │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │    Preprocessing    │
                │ Normalization       │
                │ Legal Suffixes      │
                │ Address Cleaning    │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ Blocking / TF-IDF   │
                │ Character N-Grams   │
                │ Country Partition   │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ Candidate Pairs     │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ Feature Engineering│
                │ String Similarity   │
                │ Address Similarity  │
                │ TF-IDF Similarity   │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ LightGBM Classifier │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ F₀.₅ Threshold      │
                │ Optimization        │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ Final Entity        │
                │ Matches             │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ Submission Files    │
                │ matching_results.tsv│
                │ candidate_pairs.tsv │
                └─────────────────────┘
```

---

## 📌 Key Design Principles

- **Precision-focused matching** through F_0.5 optimization.
- **Scalable candidate generation** using TF-IDF character n-grams.
- **Search-space reduction** through country-based blocking.
- **Robust entity matching** using multiple string and address similarity features.
- **Class-imbalance handling** using LightGBM.
- **Memory-efficient processing** through chunked sparse operations.
- **Conservative predictions** by applying a tuned confidence threshold.
- **Submission validation** before final packaging.

---

## 🛠️ Technology Stack

| Component | Technology |
|---|---|
| Language | Python 3.10+ |
| ML Model | LightGBM |
| Text Vectorization | TF-IDF |
| String Matching | RapidFuzz |
| Data Processing | Pandas / NumPy |
| Similarity | Levenshtein / Cosine / Jaccard |
| Evaluation | Macro F_0.5 |
| Data Format | TSV |
| Version Control | Git / GitHub |

---

## 👥 Amazon ML Challenge 2026

**Project:** Business Entity Resolution

**Objective:** Identify and link records referring to the same real-world business entity across independent data sources while maintaining high precision and efficient large-scale processing.
