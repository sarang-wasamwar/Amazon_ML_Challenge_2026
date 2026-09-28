import argparse
import gc
import os
import subprocess
import sys
import pandas as pd

from preprocess import load_tsv, preprocess_dataframe
from blocking import generate_candidates
from features import extract_pair_features
from model import (
    load_ground_truth,
    train_and_tune_threshold,
    predict_and_save_matching_results,
)


def get_all_s1_ids(s1_path: str) -> list:
    """Read first column entity_id list from Source 1 TSV efficiently."""
    df = pd.read_csv(s1_path, sep="\t", usecols=["entity_id"], dtype=str)
    return df["entity_id"].astype(str).str.strip().tolist()


def main():
    parser = argparse.ArgumentParser(
        description="Business Entity Resolution End-to-End ML Pipeline (Macro F_0.5 Optimized)"
    )
    parser.add_argument(
        "--train-dir",
        default="student_resource/dataset/train",
        help="Path to training dataset folder",
    )
    parser.add_argument(
        "--test-dir",
        default="student_resource/dataset/test",
        help="Path to test dataset folder",
    )
    parser.add_argument(
        "--output-dir",
        default="student_resource/output",
        help="Path to output directory for results",
    )
    parser.add_argument(
        "--validator-script",
        default="student_resource/utils/validate_submission.py",
        help="Path to validate_submission.py script",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=15,
        help="Top-N candidate pairs to generate per S1 record in blocking pass",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Sample size for fast prototyping / testing (default: None for full run)",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip running submission validator subprocess",
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    matching_output_path = os.path.join(args.output_dir, "matching_results.tsv")
    candidate_output_path = os.path.join(args.output_dir, "candidate_pairs.tsv")

    print("=========================================================================")
    print("      BUSINESS ENTITY RESOLUTION PIPELINE — MACRO F_0.5 OPTIMIZED        ")
    print("=========================================================================")

    # ---------------------------------------------------------
    # STEP 1: LOAD DATASETS WITH POSITIVE-AWARE SAMPLING
    # ---------------------------------------------------------
    print("\n[STEP 1/6] Loading Training and Test Datasets...")
    train_s1_path = os.path.join(args.train_dir, "train_source1.tsv")
    train_s2_path = os.path.join(args.train_dir, "train_source2.tsv")
    train_s3_path = os.path.join(args.train_dir, "train_source3.tsv")
    train_gt_path = os.path.join(args.train_dir, "train_ground_truth.tsv")

    test_s1_path = os.path.join(args.test_dir, "test_source1.tsv")
    test_s2_path = os.path.join(args.test_dir, "test_source2.tsv")
    test_s3_path = os.path.join(args.test_dir, "test_source3.tsv")

    # Always fetch complete list of test S1 IDs so final TSV includes all required rows
    all_test_s1_ids_full = get_all_s1_ids(test_s1_path)

    n_sample = args.sample_size

    if n_sample:
        print(f"[Sampling Strategy] Positive-aware sampling of top {n_sample:,} records...")
        full_gt_df = pd.read_csv(train_gt_path, sep="\t", dtype=str).fillna("")
        
        pos_gt = full_gt_df[full_gt_df["matched_entity_ids"].astype(str).str.strip() != ""]
        singleton_gt = full_gt_df[full_gt_df["matched_entity_ids"].astype(str).str.strip() == ""]

        n_pos_want = min(len(pos_gt), int(n_sample * 0.7))
        n_sing_want = min(len(singleton_gt), n_sample - n_pos_want)

        sampled_pos = pos_gt.head(n_pos_want)
        sampled_sing = singleton_gt.head(n_sing_want)

        sampled_gt = pd.concat([sampled_pos, sampled_sing], ignore_index=True)
        sampled_s1_ids = set(sampled_gt["source1_entity_id"].tolist())

        pos_s23_ids = set()
        for m_str in sampled_pos["matched_entity_ids"]:
            for m_id in str(m_str).split(","):
                if m_id.strip():
                    pos_s23_ids.add(m_id.strip())

        # Load S1 filtered by sampled IDs
        train_s1_full = load_tsv(train_s1_path)
        train_s1_raw = train_s1_full[train_s1_full["entity_id"].isin(sampled_s1_ids)].copy()
        del train_s1_full

        # Load S2 and S3 including positive match IDs
        train_s2_full = load_tsv(train_s2_path, nrows=n_sample * 3)
        train_s3_full = load_tsv(train_s3_path, nrows=n_sample * 3)

        train_s2_raw = train_s2_full.copy()
        train_s3_raw = train_s3_full.copy()

        ground_truth_dict = load_ground_truth(train_gt_path, s1_ids_filter=sampled_s1_ids)

        # Removed test_s1_raw, test_s2_raw, test_s3_raw loading to save memory until Step 5
        gc.collect()
    else:
        train_s1_raw = load_tsv(train_s1_path)
        train_s2_raw = load_tsv(train_s2_path)
        train_s3_raw = load_tsv(train_s3_path)
        ground_truth_dict = load_ground_truth(train_gt_path)

        # Removed test data loading here to save memory

    print(f"  Train S1 Records: {len(train_s1_raw):,}")
    print(f"  Train S2 Records: {len(train_s2_raw):,}")
    print(f"  Train S3 Records: {len(train_s3_raw):,}")
    # Removed Test printout here since it's loaded later

    # ---------------------------------------------------------
    # STEP 2: PREPROCESSING
    # ---------------------------------------------------------
    print("\n[STEP 2/6] Preprocessing Datasets (Text Cleaning & Address Standardization)...")
    train_s1 = preprocess_dataframe(train_s1_raw)
    train_s2 = preprocess_dataframe(train_s2_raw)
    train_s3 = preprocess_dataframe(train_s3_raw)

    del train_s1_raw, train_s2_raw, train_s3_raw
    gc.collect()

    # Removed test preprocessing here

    # ---------------------------------------------------------
    # STEP 3: CANDIDATE GENERATION (BLOCKING) — TRAIN
    # ---------------------------------------------------------
    print("\n[STEP 3/6] Running Blocking / Candidate Generation on Training Set...")
    train_pairs_df, _ = generate_candidates(
        train_s1, train_s2, train_s3, top_n=args.top_n
    )
    train_s23 = pd.concat([train_s2, train_s3], ignore_index=True)

    # ---------------------------------------------------------
    # STEP 4: FEATURE ENGINEERING & MODEL TRAINING — TRAIN
    # ---------------------------------------------------------
    print("\n[STEP 4/6] Extracting Features & Training LightGBM Model with Threshold Tuning...")
    train_features_df = extract_pair_features(train_pairs_df, train_s1, train_s23)

    del train_pairs_df, train_s2, train_s3, train_s23
    gc.collect()

    all_train_s1_ids = set(train_s1["entity_id"].tolist())
    model, tuned_tau, val_f05 = train_and_tune_threshold(
        train_features_df, ground_truth_dict, all_train_s1_ids
    )

    del train_features_df, train_s1, ground_truth_dict
    gc.collect()

    # ---------------------------------------------------------
    # STEP 5: CANDIDATE GENERATION & INFERENCE — TEST
    # ---------------------------------------------------------
    print("\n[STEP 5/6] Loading Test Sets & Preprocessing...")
    if n_sample:
        test_s1_raw = load_tsv(test_s1_path, nrows=n_sample)
        test_s2_raw = load_tsv(test_s2_path, nrows=n_sample * 3)
        test_s3_raw = load_tsv(test_s3_path, nrows=n_sample * 3)
    else:
        test_s1_raw = load_tsv(test_s1_path)
        test_s2_raw = load_tsv(test_s2_path)
        test_s3_raw = load_tsv(test_s3_path)

    test_s1 = preprocess_dataframe(test_s1_raw)
    test_s2 = preprocess_dataframe(test_s2_raw)
    test_s3 = preprocess_dataframe(test_s3_raw)
    del test_s1_raw, test_s2_raw, test_s3_raw
    gc.collect()

    print("\n[STEP 5/6] Running Candidate Generation & Model Inference on Test Set...")
    test_pairs_df, _ = generate_candidates(
        test_s1, test_s2, test_s3, top_n=args.top_n, output_path=candidate_output_path
    )
    test_s23 = pd.concat([test_s2, test_s3], ignore_index=True)

    test_features_df = extract_pair_features(test_pairs_df, test_s1, test_s23)

    del test_pairs_df, test_s2, test_s3, test_s23
    gc.collect()

    # When saving output, pass all_test_s1_ids_full so ALL 1.73M S1 entities are included in TSV
    results_df = predict_and_save_matching_results(
        model, test_features_df, tuned_tau, all_test_s1_ids_full, matching_output_path
    )

    # Also update candidate_pairs.tsv if sampled to contain all_test_s1_ids_full
    if n_sample:
        print(f"[Sampling Output Format] Padding candidate_pairs.tsv for all {len(all_test_s1_ids_full):,} test S1 entities...")
        existing_cand = pd.read_csv(candidate_output_path, sep="\t", dtype=str).fillna("")
        cand_map = dict(zip(existing_cand["source1_entity_id"], existing_cand["candidate_entity_ids"]))
        full_cand_rows = []
        for s1_id in all_test_s1_ids_full:
            full_cand_rows.append({
                "source1_entity_id": s1_id,
                "candidate_entity_ids": cand_map.get(s1_id, "")
            })
        pd.DataFrame(full_cand_rows).to_csv(candidate_output_path, sep="\t", index=False, encoding="utf-8")

    print("\nPipeline execution finished successfully!")
    print(f"  Matching Results: {matching_output_path}")
    print(f"  Candidate Pairs:  {candidate_output_path}")
    print(f"  Optimal Decision Threshold tau*: {tuned_tau:.2f}")
    print(f"  Validation Macro F_0.5 Score:   {val_f05:.4f}")

    # ---------------------------------------------------------
    # STEP 6: SUBMISSION VALIDATION
    # ---------------------------------------------------------
    if not args.skip_validation and os.path.exists(args.validator_script):
        print("\n[STEP 6/6] Executing Submission Validator Script...")
        cmd = [
            sys.executable,
            args.validator_script,
            "--matching",
            matching_output_path,
            "--candidate",
            candidate_output_path,
            "--test-dir",
            args.test_dir,
        ]
        print("Command:", " ".join(cmd))
        result = subprocess.run(cmd)
        if result.returncode == 0:
            print("\nSUCCESS: Submission output passed all formatting and structure checks!")
        else:
            print(f"\nWARNING: Validator returned exit code {result.returncode}")


if __name__ == "__main__":
    main()
