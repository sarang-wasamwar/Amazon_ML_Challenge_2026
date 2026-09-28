import os
import re
import pandas as pd


# Legal company suffixes to strip from business names
LEGAL_SUFFIXES = [
    r"\bcorporation\b",
    r"\bcorporate\b",
    r"\bincorporated\b",
    r"\blimited\b",
    r"\bprivate\b",
    r"\bcompany\b",
    r"\bcorp\b",
    r"\binc\b",
    r"\bltd\b",
    r"\bpvt\b",
    r"\bllc\b",
    r"\bpc\b",
    r"\bco\b",
]

# Combined regex pattern for legal suffix removal
SUFFIX_PATTERN = re.compile("|".join(LEGAL_SUFFIXES), re.IGNORECASE)

# Address term mappings across US, India, France
ADDRESS_MAPPINGS = [
    # US / General address terms & directionals
    (r"\brd\b", "road"),
    (r"\bst\b", "street"),
    (r"\bave\b", "avenue"),
    (r"\bav\b", "avenue"),
    (r"\bblvd\b", "boulevard"),
    (r"\bdr\b", "drive"),
    (r"\bln\b", "lane"),
    (r"\bct\b", "court"),
    (r"\bpkwy\b", "parkway"),
    (r"\bhwy\b", "highway"),
    (r"\bste\b", "suite"),
    (r"\bapt\b", "apartment"),
    (r"\bbldg\b", "building"),
    (r"\bflr?\b", "floor"),
    (r"\bn\b", "north"),
    (r"\bs\b", "south"),
    (r"\be\b", "east"),
    (r"\bw\b", "west"),
    (r"\bnw\b", "northwest"),
    (r"\bse\b", "southeast"),
    (r"\bne\b", "northeast"),
    (r"\bsw\b", "southwest"),
    # India address terms & major cities
    (r"\bopp\b", "opposite"),
    (r"\bnr\b", "near"),
    (r"\bb/?h\b", "behind"),
    (r"\bh\s*no\b", "house number"),
    (r"\bflt\b", "flat"),
    (r"\btwr\b", "tower"),
    (r"\bstn\b", "station"),
    (r"\bsec\b", "sector"),
    (r"\bind\b", "industrial"),
    (r"\best\b", "estate"),
    (r"\bsoc\b", "society"),
    (r"\bextn?\b", "extension"),
    (r"\bblr\b", "bengaluru"),
    (r"\bbangalore\b", "bengaluru"),
    (r"\bbom\b", "mumbai"),
    (r"\bbombay\b", "mumbai"),
    (r"\bcal\b", "kolkata"),
    (r"\bcalcutta\b", "kolkata"),
    (r"\bmadras\b", "chennai"),
    (r"\bhyd\b", "hyderabad"),
    # France address terms
    (r"\br\b", "rue"),
    (r"\bav\b", "avenue"),
    (r"\bbd\b", "boulevard"),
    (r"\ball\b", "allee"),
    (r"\bimp\b", "impasse"),
    (r"\bpl\b", "place"),
    (r"\bsq\b", "square"),
    (r"\bbat\b", "batiment"),
    (r"\bapt\b", "appartement"),
    (r"\bres\b", "residence"),
    (r"\bst\b", "saint"),
    (r"\bste\b", "sainte"),
]

COMPILED_ADDRESS_MAPPINGS = [
    (re.compile(pat, re.IGNORECASE), repl) for pat, repl in ADDRESS_MAPPINGS
]


def load_tsv(filepath: str, nrows: int = None) -> pd.DataFrame:
    """Load a TSV file with tab separation and string types, filling NaNs with empty string."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file not found: {filepath}")
    df = pd.read_csv(filepath, sep="\t", dtype=str, nrows=nrows).fillna("")
    return df


def clean_text(text: str) -> str:
    """Basic cleaning: lowercase, remove punctuation, collapse whitespace."""
    if not text:
        return ""
    text = text.lower()
    # Replace non-alphanumeric characters with space
    text = re.sub(r"[^\w\s]", " ", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_business_name(name_series: pd.Series) -> pd.Series:
    """Clean business names: lowercase, punctuation removal, legal suffix stripping."""
    cleaned = (
        name_series.astype(str)
        .str.lower()
        .str.replace(r"[^\w\s]", " ", regex=True)
    )
    # Strip legal suffixes
    cleaned = cleaned.str.replace(SUFFIX_PATTERN, "", regex=True)
    # Collapse multiple spaces
    cleaned = cleaned.str.replace(r"\s+", " ", regex=True).str.strip()
    return cleaned


def standardize_address(address_series: pd.Series) -> pd.Series:
    """Standardize address string series across US, India, and France."""
    cleaned = (
        address_series.astype(str)
        .str.lower()
        .str.replace(r"[^\w\s]", " ", regex=True)
    )
    for pat, repl in COMPILED_ADDRESS_MAPPINGS:
        cleaned = cleaned.str.replace(pat, repl, regex=True)
    cleaned = cleaned.str.replace(r"\s+", " ", regex=True).str.strip()
    return cleaned


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess dataframe columns: entity_id, business_name, business_address, country.

    Adds `clean_name`, `clean_address`, and `combined_text` columns.
    """
    df_out = df.copy()
    if "business_name" in df_out.columns:
        df_out["clean_name"] = clean_business_name(df_out["business_name"])
    else:
        df_out["clean_name"] = ""

    if "business_address" in df_out.columns:
        df_out["clean_address"] = standardize_address(df_out["business_address"])
    else:
        df_out["clean_address"] = ""

    df_out["country"] = df_out.get("country", pd.Series([""] * len(df_out))).astype(str).str.upper().str.strip()

    df_out["combined_text"] = (df_out["clean_name"] + " " + df_out["clean_address"]).str.strip()
    
    # Drop raw columns to save memory
    cols_to_drop = ["business_name", "business_address", "city", "state", "zip_code", "phone_number"]
    df_out.drop(columns=[c for c in cols_to_drop if c in df_out.columns], inplace=True)
    
    return df_out


if __name__ == "__main__":
    # Sanity test
    test_df = pd.DataFrame({
        "entity_id": ["S1-1", "S2-2", "S3-3"],
        "business_name": ["Acme Corp Pvt Ltd", "Prabhav Business Center", "Societe Dupont S.A.R.L."],
        "business_address": ["123 Main St, Ste 400, NY", "Nr SBI ATM, B/H Natraj Cinema, Bom", "15 R. de la Paix, Bat A, Paris"],
        "country": ["US", "India", "France"]
    })
    preprocessed = preprocess_dataframe(test_df)
    print("Preprocessed Output Sample:")
    print(preprocessed[["entity_id", "clean_name", "clean_address", "combined_text", "country"]])
