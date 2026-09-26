import gc
import os
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer


def generate_candidates(
    df_s1: pd.DataFrame,
    df_s2: pd.DataFrame,
    df_s3: pd.DataFrame,
    top_n: int = 15,
    sub_batch_size: int = 500,
    min_similarity: float = 0.25,
    output_path: str = None,
) -> tuple:
    """Generate top-N candidate matches per S1 entity using memory-efficient partitioned TF-IDF cosine similarity.

    Memory & Speed Optimizations:
    - Country-based partitioning (e.g. US against US, India against India).
    - Sub-batching (500 S1 records at a time).
    - Sparse float32 arithmetic.
    - Zero-copy CSR sparse slicing & score pruning (< min_similarity).
    - Explicit garbage collection per batch.
    """
    print(f"[Blocking] Starting candidate generation for {len(df_s1):,} S1 records...")

    # Combine Source 2 and Source 3
    df_s23 = pd.concat([df_s2, df_s3], ignore_index=True)

    # Pre-clean country column strings
    df_s1 = df_s1.copy()
    df_s23 = df_s23.copy()
    df_s1["country_clean"] = df_s1["country"].astype(str).str.upper().str.strip()
    df_s23["country_clean"] = df_s23["country"].astype(str).str.upper().str.strip()

    s1_ids_all = df_s1["entity_id"].values
    candidate_dict = {s1_id: [] for s1_id in s1_ids_all}
    pair_rows = []

    # Get distinct countries
    countries = np.unique(df_s1["country_clean"])

    for country in countries:
        s1_country_mask = df_s1["country_clean"] == country
        s23_country_mask = df_s23["country_clean"] == country

        # Fallback to full pool if country has no S23 records
        if not s23_country_mask.any():
            s23_country_mask = np.ones(len(df_s23), dtype=bool)

        df_s1_sub = df_s1[s1_country_mask].reset_index(drop=True)
        df_s23_sub = df_s23[s23_country_mask].reset_index(drop=True)

        n_s1_sub = len(df_s1_sub)
        n_s23_sub = len(df_s23_sub)

        print(f"[Blocking] Country '{country}': {n_s1_sub:,} S1 vs {n_s23_sub:,} S2/S3 records...")

        # Build character n-gram TF-IDF vectorizer per country partition
        vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 4),
            min_df=1,
            sublinear_tf=True,
            max_features=100000,
            dtype=np.float32,
        )

        all_texts = pd.concat(
            [df_s1_sub["combined_text"], df_s23_sub["combined_text"]],
            ignore_index=True,
        )
        vectorizer.fit(all_texts)

        X_s1_ctry = vectorizer.transform(df_s1_sub["combined_text"]).astype(np.float32)
        X_s23_ctry = vectorizer.transform(df_s23_sub["combined_text"]).astype(np.float32)

        s1_sub_ids = df_s1_sub["entity_id"].values
        s23_sub_ids = df_s23_sub["entity_id"].values

        # Process in small sub-batches to bound memory footprint
        for start_idx in range(0, n_s1_sub, sub_batch_size):
            end_idx = min(start_idx + sub_batch_size, n_s1_sub)
            X_batch = X_s1_ctry[start_idx:end_idx]

            # Sparse matrix dot product
            sim_sub = X_batch.dot(X_s23_ctry.T).tocsr()

            # Prune similarity scores below threshold
            sim_sub.data[sim_sub.data < min_similarity] = 0.0
            sim_sub.eliminate_zeros()

            indptr = sim_sub.indptr
            indices = sim_sub.indices
            data = sim_sub.data

            for row_in_batch in range(end_idx - start_idx):
                s1_id = s1_sub_ids[start_idx + row_in_batch]
                
                row_start = indptr[row_in_batch]
                row_end = indptr[row_in_batch + 1]

                if row_start == row_end:
                    continue

                row_cols = indices[row_start:row_end]
                row_vals = data[row_start:row_end]

                n_elems = len(row_vals)
                if n_elems > top_n:
                    top_k_arg = np.argpartition(row_vals, -top_n)[-top_n:]
                    top_k_arg = top_k_arg[np.argsort(-row_vals[top_k_arg])]
                    top_cand_indices = row_cols[top_k_arg]
                    top_sims = row_vals[top_k_arg]
                else:
                    order = np.argsort(-row_vals)
                    top_cand_indices = row_cols[order]
                    top_sims = row_vals[order]

                cand_ids = s23_sub_ids[top_cand_indices].tolist()
                candidate_dict[s1_id] = cand_ids

                for c_id, sim_score in zip(cand_ids, top_sims):
                    pair_rows.append({
                        "source1_entity_id": s1_id,
                        "candidate_entity_id": c_id,
                        "tfidf_sim": float(sim_score),
                    })

            del sim_sub, X_batch
            gc.collect()

        del X_s1_ctry, X_s23_ctry, vectorizer, all_texts, df_s1_sub, df_s23_sub
        gc.collect()

    pairs_df = pd.DataFrame(pair_rows)
    print(f"[Blocking] Completed candidate pair generation. Total pairs: {len(pairs_df):,}")

    # Save to candidate_pairs.tsv if output_path is provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        tsv_rows = []
        for s1_id in s1_ids_all:
            c_list = candidate_dict.get(s1_id, [])
            c_str = ",".join(c_list) if c_list else ""
            tsv_rows.append({"source1_entity_id": s1_id, "candidate_entity_ids": c_str})

        cand_pairs_tsv_df = pd.DataFrame(tsv_rows)
        cand_pairs_tsv_df.to_csv(output_path, sep="\t", index=False, encoding="utf-8")
        print(f"[Blocking] Saved candidate pairs to {output_path} ({len(cand_pairs_tsv_df):,} S1 rows)")

    return pairs_df, candidate_dict


if __name__ == "__main__":
    from preprocess import preprocess_dataframe

    s1 = preprocess_dataframe(
        pd.DataFrame({
            "entity_id": ["S1-1", "S1-2"],
            "business_name": ["Acme Corp", "Beta Solutions"],
            "business_address": ["123 Main St NY", "456 Market St CA"],
            "country": ["US", "US"],
        })
    )
    s2 = preprocess_dataframe(
        pd.DataFrame({
            "entity_id": ["S2-10", "S2-20"],
            "business_name": ["Acme Incorporated", "Beta Tech"],
            "business_address": [
                "123 Main Street New York",
                "456 Market St San Francisco",
            ],
            "country": ["US", "US"],
        })
    )
    s3 = preprocess_dataframe(
        pd.DataFrame({
            "entity_id": ["S3-100"],
            "business_name": ["Acme Company"],
            "business_address": ["123 Main St Suite 1 NY"],
            "country": ["US"],
        })
    )
    pairs_df, cand_dict = generate_candidates(
        s1, s2, s3, top_n=5, output_path="student_resource/output/test_cand.tsv"
    )
    print("Candidate Pairs DF:")
    print(pairs_df)
    print("Candidate Dict:")
    print(cand_dict)
