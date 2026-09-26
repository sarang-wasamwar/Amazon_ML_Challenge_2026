import os
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def load_ground_truth(gt_filepath: str, s1_ids_filter: set = None) -> dict:
    """Load ground truth TSV into dictionary: source1_entity_id -> set of matched_entity_ids."""
    gt_df = pd.read_csv(gt_filepath, sep="\t", dtype=str).fillna("")
    if s1_ids_filter is not None:
        gt_df = gt_df[gt_df["source1_entity_id"].isin(s1_ids_filter)]
    gt_dict = {}
    for row in gt_df.itertuples(index=False):
        s1_id = str(row.source1_entity_id).strip()
        m_str = str(row.matched_entity_ids).strip()
        if m_str:
            matched_ids = set([x.strip() for x in m_str.split(",") if x.strip()])
        else:
            matched_ids = set()
        gt_dict[s1_id] = matched_ids
    return gt_dict


def compute_macro_f05(predictions: dict, ground_truth: dict, all_s1_ids: set) -> float:
    """Calculate Macro F_0.5 score across all Source 1 entities (including singletons).

    Formula per entity:
    F_0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)
    Singletons:
      - true_set empty & pred_set empty => F_0.5 = 1.0
      - true_set empty & pred_set non-empty => F_0.5 = 0.0
    """
    total_f05 = 0.0
    n = len(all_s1_ids)
    if n == 0:
        return 0.0

    for s1_id in all_s1_ids:
        pred_set = predictions.get(s1_id, set())
        true_set = ground_truth.get(s1_id, set())

        if len(true_set) == 0:
            if len(pred_set) == 0:
                f05 = 1.0
            else:
                f05 = 0.0
        else:
            if len(pred_set) == 0:
                f05 = 0.0
            else:
                tp = len(pred_set & true_set)
                precision = tp / float(len(pred_set))
                recall = tp / float(len(true_set))

                denom = (0.25 * precision) + recall
                if denom > 0.0:
                    f05 = (1.25 * precision * recall) / denom
                else:
                    f05 = 0.0

        total_f05 += f05

    return total_f05 / float(n)


def train_and_tune_threshold(
    features_df: pd.DataFrame,
    ground_truth: dict,
    all_train_s1_ids: set,
) -> tuple:
    """Train LightGBM Classifier with class imbalance handling and tune decision threshold tau to maximize Macro F_0.5.

    Returns:
    --------
    tuple : (model, best_tau, best_val_f05)
    """
    print("[Model] Constructing training target labels from ground truth...")

    # Create binary target labels: 1 if candidate in ground truth, else 0
    labels = []
    for _, row in features_df.iterrows():
        s1_id = row["source1_entity_id"]
        c_id = row["candidate_entity_id"]
        true_matches = ground_truth.get(s1_id, set())
        labels.append(1 if c_id in true_matches else 0)

    features_df = features_df.copy()
    features_df["label"] = labels

    feature_cols = [
        c
        for c in features_df.columns
        if c not in ["source1_entity_id", "candidate_entity_id", "label"]
    ]

    n_positives = sum(labels)
    print(f"[Model] Total candidate pairs for training: {len(features_df):,} (Positives: {n_positives:,})")

    # Train / Val entity-level split
    train_s1_ids, val_s1_ids = train_test_split(
        list(all_train_s1_ids), test_size=0.20, random_state=42
    )
    val_s1_set = set(val_s1_ids)

    train_mask = features_df["source1_entity_id"].isin(train_s1_ids)
    val_mask = features_df["source1_entity_id"].isin(val_s1_set)

    X_train = features_df.loc[train_mask, feature_cols]
    y_train = features_df.loc[train_mask, "label"]

    X_val = features_df.loc[val_mask, feature_cols]
    y_val = features_df.loc[val_mask, "label"]

    val_pairs = features_df.loc[val_mask, ["source1_entity_id", "candidate_entity_id"]]

    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)

    if len(np.unique(y_train)) < 2:
        print("[Model Warning] Target contains only 1 class in training set. Fitting baseline model without tree splits...")
        model = lgb.LGBMClassifier(
            n_estimators=400,
            learning_rate=0.05,
            num_leaves=31,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            is_unbalance=True,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
        model.fit(X_train, y_train)
        return model, 0.70, 0.0

    print("[Model] Attempting LightGBM GPU Acceleration (device='gpu', platform=0, device=0)...")
    try:
        model = lgb.LGBMClassifier(
            n_estimators=400,
            learning_rate=0.05,
            num_leaves=31,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            is_unbalance=True,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
            device="gpu",
            gpu_platform_id=0,
            gpu_device_id=0,
        )
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
        )
        print("[Model GPU] Successfully trained LightGBM model using GPU acceleration!")
    except Exception as exc:
        print(f"[Model Warning] GPU acceleration unavailable ({exc}). Falling back to CPU mode...")
        model = lgb.LGBMClassifier(
            n_estimators=400,
            learning_rate=0.05,
            num_leaves=31,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            is_unbalance=True,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
        )

    # Decision threshold tuning on validation set
    val_probs = model.predict_proba(X_val)[:, 1]
    val_pairs = val_pairs.copy()
    val_pairs["prob"] = val_probs

    best_tau = 0.70
    best_val_f05 = -1.0
    threshold_grid = np.arange(0.60, 0.86, 0.01)

    print("[Model] Tuning decision threshold tau for Macro F_0.5 optimization...")
    for tau in threshold_grid:
        tau = round(float(tau), 2)
        # Filter predictions at current threshold
        matched_pairs = val_pairs[val_pairs["prob"] >= tau]

        pred_dict = {}
        for s1_id, group in matched_pairs.groupby("source1_entity_id"):
            pred_dict[s1_id] = set(group["candidate_entity_id"].tolist())

        f05_score = compute_macro_f05(pred_dict, ground_truth, val_s1_set)
        if f05_score > best_val_f05:
            best_val_f05 = f05_score
            best_tau = tau

    print(f"[Model] Best Threshold tau* = {best_tau:.2f} | Max Validation Macro F_0.5 = {best_val_f05:.4f}")
    return model, best_tau, best_val_f05


def predict_and_save_matching_results(
    model,
    test_features_df: pd.DataFrame,
    tau: float,
    all_test_s1_ids: list,
    output_path: str,
) -> pd.DataFrame:
    """Predict matches on test set using tuned threshold tau and write matching_results.tsv."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    test_pred_dict = {}

    if not test_features_df.empty:
        feature_cols = [
            c
            for c in test_features_df.columns
            if c not in ["source1_entity_id", "candidate_entity_id", "label"]
        ]
        test_probs = model.predict_proba(test_features_df[feature_cols])[:, 1]
        
        test_df = test_features_df[["source1_entity_id", "candidate_entity_id"]].copy()
        test_df["prob"] = test_probs

        matched_test = test_df[test_df["prob"] >= tau]

        for s1_id, group in matched_test.groupby("source1_entity_id"):
            test_pred_dict[s1_id] = group["candidate_entity_id"].tolist()

    out_rows = []
    for s1_id in all_test_s1_ids:
        matches = test_pred_dict.get(s1_id, [])
        match_str = ",".join(matches) if matches else ""
        out_rows.append({"source1_entity_id": s1_id, "matched_entity_ids": match_str})

    results_df = pd.DataFrame(out_rows)
    results_df.to_csv(output_path, sep="\t", index=False, encoding="utf-8")
    print(f"[Inference] Saved matching results to {output_path} ({len(results_df):,} S1 rows)")
    return results_df


if __name__ == "__main__":
    gt = {"S1-1": {"S2-10", "S3-100"}, "S1-2": set()}
    preds = {"S1-1": {"S2-10", "S3-100"}, "S1-2": set()}
    score = compute_macro_f05(preds, gt, {"S1-1", "S1-2"})
    print("Test Macro F0.5 Score:", score)
