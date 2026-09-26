import re
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein


def extract_pair_features(
    pairs_df: pd.DataFrame,
    df_s1: pd.DataFrame,
    df_s23: pd.DataFrame,
) -> pd.DataFrame:
    """Compute similarity features for every candidate pair.

    Features computed:
    - name_levenshtein: Levenshtein normalized similarity on clean_name
    - name_token_sort: RapidFuzz token_sort_ratio on clean_name
    - name_token_set: RapidFuzz token_set_ratio on clean_name
    - address_levenshtein: Levenshtein normalized similarity on clean_address
    - address_token_sort: RapidFuzz token_sort_ratio on clean_address
    - address_token_set: RapidFuzz token_set_ratio on clean_address
    - address_jaccard: Token Jaccard similarity on clean_address
    - combined_token_set: RapidFuzz token_set_ratio on combined_text
    - tfidf_cosine_sim: TF-IDF cosine similarity from blocking
    - digit_match_count: Count of matching digit strings between records
    - digit_jaccard: Jaccard similarity of extracted digit sets
    - exact_name_match: Binary flag (1.0 if clean names match exactly)
    - exact_address_match: Binary flag (1.0 if clean addresses match exactly)
    - country_match: Binary flag (1.0 if countries match)
    """
    if pairs_df.empty:
        return pd.DataFrame()

    print(f"[Feature Engineering] Extracting features for {len(pairs_df)} candidate pairs...")

    # Build efficient lookup dictionaries
    s1_dict = df_s1.set_index("entity_id")[
        ["clean_name", "clean_address", "combined_text", "country"]
    ].to_dict("index")
    
    s23_dict = df_s23.set_index("entity_id")[
        ["clean_name", "clean_address", "combined_text", "country"]
    ].to_dict("index")

    digit_pattern = re.compile(r"\d+")

    feature_rows = []

    s1_ids = pairs_df["source1_entity_id"].values
    cand_ids = pairs_df["candidate_entity_id"].values
    tfidf_sims = pairs_df.get("tfidf_sim", pd.Series([0.0] * len(pairs_df))).values

    for s1_id, c_id, tfidf_sim in zip(s1_ids, cand_ids, tfidf_sims):
        s1_rec = s1_dict.get(s1_id, {"clean_name": "", "clean_address": "", "combined_text": "", "country": ""})
        c_rec = s23_dict.get(c_id, {"clean_name": "", "clean_address": "", "combined_text": "", "country": ""})

        s1_name, c_name = s1_rec["clean_name"], c_rec["clean_name"]
        s1_addr, c_addr = s1_rec["clean_address"], c_rec["clean_address"]
        s1_comb, c_comb = s1_rec["combined_text"], c_rec["combined_text"]
        s1_ctry, c_ctry = s1_rec["country"], c_rec["country"]

        # Levenshtein ratio
        name_lev = Levenshtein.normalized_similarity(s1_name, c_name)
        addr_lev = Levenshtein.normalized_similarity(s1_addr, c_addr)

        # RapidFuzz token_sort and token_set ratios
        name_tsort = fuzz.token_sort_ratio(s1_name, c_name) / 100.0
        name_tset = fuzz.token_set_ratio(s1_name, c_name) / 100.0

        addr_tsort = fuzz.token_sort_ratio(s1_addr, c_addr) / 100.0
        addr_tset = fuzz.token_set_ratio(s1_addr, c_addr) / 100.0

        comb_tset = fuzz.token_set_ratio(s1_comb, c_comb) / 100.0

        # Address token Jaccard similarity
        s1_addr_tokens = set(s1_addr.split())
        c_addr_tokens = set(c_addr.split())
        if s1_addr_tokens or c_addr_tokens:
            addr_jaccard = len(s1_addr_tokens & c_addr_tokens) / float(len(s1_addr_tokens | c_addr_tokens))
        else:
            addr_jaccard = 1.0 if s1_addr == c_addr else 0.0

        # Digit match count & digit Jaccard
        s1_digits = set(digit_pattern.findall(s1_comb))
        c_digits = set(digit_pattern.findall(c_comb))
        common_digits = s1_digits & c_digits
        digit_match_count = float(len(common_digits))

        if s1_digits or c_digits:
            digit_jaccard = len(common_digits) / float(len(s1_digits | c_digits))
        else:
            digit_jaccard = 1.0

        # Match indicators
        exact_name_match = 1.0 if (s1_name and s1_name == c_name) else 0.0
        exact_address_match = 1.0 if (s1_addr and s1_addr == c_addr) else 0.0
        country_match = 1.0 if (s1_ctry and s1_ctry == c_ctry) else 0.0

        feature_rows.append({
            "source1_entity_id": s1_id,
            "candidate_entity_id": c_id,
            "name_levenshtein": name_lev,
            "name_token_sort": name_tsort,
            "name_token_set": name_tset,
            "address_levenshtein": addr_lev,
            "address_token_sort": addr_tsort,
            "address_token_set": addr_tset,
            "address_jaccard": addr_jaccard,
            "combined_token_set": comb_tset,
            "tfidf_cosine_sim": tfidf_sim,
            "digit_match_count": digit_match_count,
            "digit_jaccard": digit_jaccard,
            "exact_name_match": exact_name_match,
            "exact_address_match": exact_address_match,
            "country_match": country_match,
        })

    features_df = pd.DataFrame(feature_rows)
    print(f"[Feature Engineering] Completed feature matrix extraction of shape {features_df.shape}.")
    return features_df


if __name__ == "__main__":
    from preprocess import preprocess_dataframe
    s1 = preprocess_dataframe(pd.DataFrame({
        "entity_id": ["S1-1"],
        "business_name": ["Acme Corp"],
        "business_address": ["123 Main St NY 10001"],
        "country": ["US"]
    }))
    s23 = preprocess_dataframe(pd.DataFrame({
        "entity_id": ["S2-10"],
        "business_name": ["Acme Inc"],
        "business_address": ["123 Main Street New York 10001"],
        "country": ["US"]
    }))
    pairs = pd.DataFrame([{"source1_entity_id": "S1-1", "candidate_entity_id": "S2-10", "tfidf_sim": 0.85}])
    feats = extract_pair_features(pairs, s1, s23)
    print(feats)
