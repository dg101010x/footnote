"""Fiscal-year windows, the test lock, and case groups.

Rule 1 of the build spec: until the freeze, rows from fiscal years 2003 to 2014
may be used only for label counts, row counts, missing-value counts and schema
checks. This module enforces that. load_split("test") and
load_split("confirmation") raise LockedSplitError unless artifacts/frozen.json
exists, and assert_unlocked() is called by every function that reads features.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from .data import CASE_ID, KEYS, LABEL, REPO_ROOT

WINDOWS: dict[str, tuple[int, int]] = {
    "select_train": (1990, 1999),
    "validation": (2000, 2001),
    "final_fit": (1990, 2001),
    "gap": (2002, 2002),
    "test": (2003, 2008),
    "confirmation": (2009, 2014),
}
# The five non-overlapping windows reported in the window table.
TABLE_WINDOWS = ["select_train", "validation", "gap", "test", "confirmation"]

LOCKED_FROM_YEAR = 2003
LOCKED_WINDOWS = {"test", "confirmation"}
SEEN_LAST_YEAR = 2001  # seen case = has a positive year in 1990 to 2001
VALIDATION_CASE_START = 2000  # validation positives from cases starting before this are dropped

FROZEN_PATH = REPO_ROOT / "artifacts" / "frozen.json"

LABEL_COLUMNS = KEYS + [LABEL, CASE_ID, "row_id", "tie_key"]


class LockedSplitError(RuntimeError):
    """Raised when locked fiscal years (2003 to 2014) are touched before the freeze."""


def frozen_path() -> Path:
    """Path of the freeze file. FOOTNOTE_FROZEN_PATH overrides it (used by tests)."""
    return Path(os.environ.get("FOOTNOTE_FROZEN_PATH", FROZEN_PATH))


def is_frozen() -> bool:
    return frozen_path().exists()


def assert_unlocked(df_or_years) -> None:
    """Raise LockedSplitError if any row is from 2003 or later and there is no freeze."""
    if is_frozen():
        return
    years = df_or_years["fyear"] if isinstance(df_or_years, pd.DataFrame) else pd.Series(df_or_years)
    if len(years) and int(pd.Series(years).max()) >= LOCKED_FROM_YEAR:
        raise LockedSplitError(
            "Fiscal years 2003 to 2014 are locked until artifacts/frozen.json exists. "
            "Only label, row, missing-value and schema counts are allowed on them."
        )


def in_window(df: pd.DataFrame, name: str) -> pd.Series:
    lo, hi = WINDOWS[name]
    return df["fyear"].between(lo, hi)


def load_split(name: str, df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of the rows (all columns) for one window.

    Raises LockedSplitError for test and confirmation unless frozen.
    """
    if name not in WINDOWS:
        raise KeyError(f"Unknown window {name!r}. Choose from {list(WINDOWS)}")
    if name in LOCKED_WINDOWS and not is_frozen():
        raise LockedSplitError(
            f"load_split({name!r}) is locked until artifacts/frozen.json exists."
        )
    out = df.loc[in_window(df, name)].copy()
    assert_unlocked(out)
    return out


def label_view(df: pd.DataFrame) -> pd.DataFrame:
    """Keys, label, case id, row id and tie key for every year. No feature columns.

    This is the only allowed view of 2003 to 2014 rows before the freeze.
    """
    return df[LABEL_COLUMNS].copy()


def case_table(labels: pd.DataFrame) -> pd.DataFrame:
    """One row per case (p_aaer) with first and last positive year, counts and seen flag."""
    pos = labels.loc[labels[LABEL] == 1]
    g = pos.groupby(CASE_ID)
    out = pd.DataFrame(
        {
            "first_year": g["fyear"].min(),
            "last_year": g["fyear"].max(),
            "n_positive_years": g.size(),
            "n_companies": g["gvkey"].nunique(),
        }
    )
    out["seen"] = out["first_year"] <= SEEN_LAST_YEAR
    return out


def case_group(labels: pd.DataFrame) -> pd.Series:
    """Per row: "seen" or "unseen" for positive rows, "" for negatives.

    Seen = the case has a positive year in 1990 to 2001.
    Unseen = the case's first positive year is 2002 or later.
    """
    first = case_table(labels)["first_year"]
    first_by_row = labels[CASE_ID].map(first)
    grp = pd.Series("", index=labels.index, dtype=object)
    pos = labels[LABEL] == 1
    grp[pos & (first_by_row <= SEEN_LAST_YEAR)] = "seen"
    grp[pos & (first_by_row > SEEN_LAST_YEAR)] = "unseen"
    return grp


def window_summary(labels: pd.DataFrame) -> pd.DataFrame:
    """Rows, positive years, companies and cases for each reported window."""
    rows = []
    for name in TABLE_WINDOWS:
        lo, hi = WINDOWS[name]
        s = labels.loc[labels["fyear"].between(lo, hi)]
        p = s.loc[s[LABEL] == 1]
        rows.append(
            {
                "window": name,
                "fiscal_years": f"{lo} to {hi}" if lo != hi else str(lo),
                "rows": len(s),
                "positive_years": len(p),
                "companies": s["gvkey"].nunique(),
                "cases": p[CASE_ID].nunique(),
            }
        )
    return pd.DataFrame(rows).set_index("window")


def validation_eval_mask(labels_val: pd.DataFrame, all_labels: pd.DataFrame) -> pd.Series:
    """True for validation rows kept for model selection.

    Drops positive rows whose case began (first positive year) before 2000.
    Case start years come from all labels, which is a label count only.
    """
    first = case_table(all_labels)["first_year"]
    began = labels_val[CASE_ID].map(first)
    drop = (labels_val[LABEL] == 1) & (began < VALIDATION_CASE_START)
    return ~drop


def seen_companies(labels: pd.DataFrame) -> set:
    """Companies with any row in 1990 to 2001."""
    return set(labels.loc[labels["fyear"] <= SEEN_LAST_YEAR, "gvkey"].unique())


def gate_a_facts(df: pd.DataFrame) -> dict:
    """The Gate A numbers. Uses labels for every year, and only total assets and
    sales counts (a missing/schema-style count) over all rows."""
    labels = label_view(df)
    cases = case_table(labels)
    grp = case_group(labels)
    val = labels.loc[in_window(labels, "validation")]
    test = in_window(labels, "test") & (labels[LABEL] == 1)
    seen_test = test & (grp == "seen")
    return {
        "validation_pos_from_cases_before_2000": int((~validation_eval_mask(val, labels)).sum()),
        "test_positive_years": int(test.sum()),
        "test_seen_positive_years": int(seen_test.sum()),
        "test_unseen_positive_years": int((test & (grp == "unseen")).sum()),
        "test_seen_cases": int(labels.loc[seen_test, CASE_ID].nunique()),
        "rows_at_le_0": int((df["at"] <= 0).sum()),
        "rows_sale_le_0": int((df["sale"] <= 0).sum()),
        "two_company_cases": int((cases["n_companies"] > 1).sum()),
    }
