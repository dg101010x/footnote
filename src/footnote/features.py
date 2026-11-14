"""The 61 model features. One function, one fiscal year, one company per row.

No feature uses another year, another company or the label. Undefined values
are stored as NaN, never zero and never infinity.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import DOLLAR_ITEMS, RAW_ITEMS, REPO_ROOT
from .splits import assert_unlocked

SCALED_ITEMS = [c for c in DOLLAR_ITEMS if c != "at"]  # 25
SCALED_FEATURES = [f"{c}_at" for c in SCALED_ITEMS]
RATIO_FEATURES = [
    "current_ratio",
    "liab_to_assets",
    "gross_margin",
    "rec_to_sales",
    "inv_to_sales",
    "soft_assets_share",
    "book_to_market",
    "log_assets",
]
FEATURE_NAMES = RAW_ITEMS + SCALED_FEATURES + RATIO_FEATURES  # 28 + 25 + 8 = 61
LOGIT_FEATURES = SCALED_FEATURES + RATIO_FEATURES  # 33, for the logistic baseline

SCHEMA_PATH = REPO_ROOT / "artifacts" / "feature_schema.json"


def _safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    """num / den where den > 0, else NaN."""
    num = num.astype("float64")
    den = den.astype("float64")
    out = num / den.where(den > 0)
    return out.where(np.isfinite(out))


def compute_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Compute the 61 features from the 28 raw items.

    Works on any frame holding the 28 raw columns, including a single
    user-entered row in the app. Row-wise only. If the frame has an fyear
    column, the test lock is checked.
    """
    if "fyear" in raw:
        assert_unlocked(raw)
    missing = [c for c in RAW_ITEMS if c not in raw]
    if missing:
        raise KeyError(f"Missing raw items {missing}")
    x = raw[RAW_ITEMS].astype("float64")
    out = {c: x[c] for c in RAW_ITEMS}
    at = x["at"]
    for c in SCALED_ITEMS:
        out[f"{c}_at"] = _safe_div(x[c], at)
    out["current_ratio"] = _safe_div(x["act"], x["lct"])
    out["liab_to_assets"] = _safe_div(x["lt"], at)
    out["gross_margin"] = _safe_div(x["sale"] - x["cogs"], x["sale"])
    out["rec_to_sales"] = _safe_div(x["rect"], x["sale"])
    out["inv_to_sales"] = _safe_div(x["invt"], x["sale"])
    out["soft_assets_share"] = _safe_div(at - x["ppegt"] - x["che"], at)
    out["book_to_market"] = _safe_div(x["ceq"], x["prcc_f"] * x["csho"])
    out["log_assets"] = np.log1p(at.clip(lower=0))
    feats = pd.DataFrame(out, index=raw.index)[FEATURE_NAMES]
    return feats


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features for rows of the company-year file. Refuses locked years before the freeze."""
    if "fyear" not in df:
        raise KeyError("build_features needs the fyear column so the lock can be checked")
    return compute_features(df)


def write_feature_schema(path: Path | str = SCHEMA_PATH) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    schema = {
        "n_features": len(FEATURE_NAMES),
        "features": FEATURE_NAMES,
        "raw_items": RAW_ITEMS,
        "scaled_features": SCALED_FEATURES,
        "ratio_features": RATIO_FEATURES,
        "logistic_features": LOGIT_FEATURES,
        "units": {
            "dollar_items": "millions of US dollars",
            "csho": "millions of shares",
            "prcc_f": "US dollars per share at fiscal year end",
        },
        "definitions": {
            "<item>_at": "item / at, missing when at <= 0",
            "current_ratio": "act / lct, missing when lct <= 0",
            "liab_to_assets": "lt / at, missing when at <= 0",
            "gross_margin": "(sale - cogs) / sale, missing when sale <= 0",
            "rec_to_sales": "rect / sale, missing when sale <= 0",
            "inv_to_sales": "invt / sale, missing when sale <= 0",
            "soft_assets_share": "(at - ppegt - che) / at, missing when at <= 0; uses gross plant; not the file's soft_assets column",
            "book_to_market": "ceq / (prcc_f * csho), missing when prcc_f * csho <= 0",
            "log_assets": "ln(1 + max(at, 0)), never missing",
        },
    }
    path.write_text(json.dumps(schema, indent=2) + "\n")
    return path
