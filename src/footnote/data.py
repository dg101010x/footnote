"""Download, verify and load the Bao et al. (2020) company-year file.

The file is never committed. It is downloaded at run time and refused unless its
size and SHA-256 match the values below.

Citation: Bao, Ke, Li, Yu and Zhang (2020), Journal of Accounting Research 58(1), 199 to 235.
"""
from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_URL = (
    "https://raw.githubusercontent.com/JarFraud/FraudDetection/master/"
    "data_FraudDetection_JAR2020.csv"
)
FILE_NAME = "data_FraudDetection_JAR2020.csv"
DEFAULT_PATH = REPO_ROOT / "data" / "raw" / FILE_NAME
EXPECTED_BYTES = 47_782_071
EXPECTED_SHA256 = "ed4cefd9db2a23d0a13911d9f482fc755d3dcf6c9be0bc051af95ddcae5b286f"
EXPECTED_ROWS = 146_045
EXPECTED_COLS = 46

KEYS = ["fyear", "gvkey"]
LABEL = "misstate"
CASE_ID = "p_aaer"

# 26 dollar items, millions of US dollars.
DOLLAR_ITEMS = [
    "act", "ap", "at", "ceq", "che", "cogs", "dlc", "dltis", "dltt", "dp",
    "ib", "invt", "ivao", "ivst", "lct", "lt", "ni", "ppegt", "pstk", "re",
    "rect", "sale", "sstk", "txp", "txt", "xint",
]
# csho is millions of shares. prcc_f is dollars per share.
NON_DOLLAR_ITEMS = ["csho", "prcc_f"]
RAW_ITEMS = DOLLAR_ITEMS + NON_DOLLAR_ITEMS  # the 28 inputs

# 14 ready-made ratios in the file. Most use the prior year. Never used by the
# primary model or the fair rules.
READY_RATIOS = [
    "dch_wc", "ch_rsst", "dch_rec", "dch_inv", "soft_assets", "ch_cs", "ch_cm",
    "ch_roa", "issue", "bm", "dpi", "reoa", "EBIT", "ch_fcf",
]

TIE_SEED = 0


class ChecksumError(ValueError):
    """Raised when the data file does not match the expected size or SHA-256."""


def sha256_file(path: Path | str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def verify_file(path: Path | str) -> None:
    """Raise ChecksumError unless the file has the expected size and SHA-256."""
    path = Path(path)
    size = path.stat().st_size
    if size != EXPECTED_BYTES:
        raise ChecksumError(
            f"{path} has {size:,} bytes, expected {EXPECTED_BYTES:,}. Refusing to load."
        )
    digest = sha256_file(path)
    if digest != EXPECTED_SHA256:
        raise ChecksumError(
            f"{path} has SHA-256 {digest}, expected {EXPECTED_SHA256}. Refusing to load."
        )


def download(path: Path | str = DEFAULT_PATH, url: str = DATA_URL) -> Path:
    """Download the file to path (via a temp file) and verify it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    try:
        verify_file(tmp)
    except ChecksumError:
        tmp.unlink(missing_ok=True)
        raise
    tmp.replace(path)
    return path


def ensure_file(path: Path | str = DEFAULT_PATH, allow_download: bool = True) -> Path:
    path = Path(path)
    if not path.exists():
        if not allow_download:
            raise FileNotFoundError(f"{path} not found and download disabled.")
        download(path)
    verify_file(path)
    return path


def check_schema(df: pd.DataFrame) -> None:
    """Schema checks that are allowed on every year (no feature values are read
    except to count missing values in the 28 inputs)."""
    if df.shape != (EXPECTED_ROWS, EXPECTED_COLS):
        raise ValueError(f"Unexpected shape {df.shape}")
    missing = [c for c in KEYS + [LABEL, CASE_ID] + RAW_ITEMS + READY_RATIOS if c not in df]
    if missing:
        raise ValueError(f"Missing columns {missing}")
    if df.duplicated(KEYS).any():
        raise ValueError("(gvkey, fyear) repeats")
    if df[RAW_ITEMS].isna().any().any():
        raise ValueError("The 28 raw inputs should have no missing values")
    if not set(df[LABEL].unique()) <= {0, 1}:
        raise ValueError("misstate must be 0 or 1")
    pos = df[LABEL] == 1
    if df.loc[pos, CASE_ID].isna().any() or df.loc[~pos, CASE_ID].notna().any():
        raise ValueError("p_aaer should be present exactly on positive rows")


def load_raw(path: Path | str = DEFAULT_PATH, allow_download: bool = True) -> pd.DataFrame:
    """Load the verified file, all years, with row_id and tie_key added.

    Do not use this directly for modelling. Use splits.load_split, which locks
    fiscal years 2003 to 2014 until artifacts/frozen.json exists.

    row_id is the 0-based position in the file. tie_key is a fixed seeded
    shuffle of row ids (numpy default_rng(0)), used only to break score ties so
    that file order never decides a flag.
    """
    path = ensure_file(path, allow_download=allow_download)
    df = pd.read_csv(path)
    check_schema(df)
    df["gvkey"] = df["gvkey"].astype("int64")
    df["fyear"] = df["fyear"].astype("int64")
    df["misstate"] = df["misstate"].astype("int64")
    df["row_id"] = np.arange(len(df), dtype=np.int64)
    rng = np.random.default_rng(TIE_SEED)
    df["tie_key"] = rng.permutation(len(df)).astype(np.int64)
    return df
