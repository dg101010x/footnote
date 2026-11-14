"""Budget flags and metrics for ranked scores (build spec sections 6 and 8).

Ranking order everywhere: score descending, missing scores last, then tie_key
ascending. Each fiscal year y flags its top K_y = ceil(b * N_y) rows. Budgets are
never pooled across years. A score is a ranking, not a probability.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from .splits import assert_unlocked

DEFAULT_BUDGETS: tuple[float, ...] = (0.01, 0.05, 0.10)
CURVE_BUDGETS: np.ndarray = np.round(np.arange(0.01, 0.2001, 0.01), 2)


def budget_k(n: int, b: float) -> int:
    """Rows reviewed for a year of n filings at budget b: ceil(b * n), float safe."""
    return int(np.ceil(np.round(b * n, 9)))


def budget_key(b: float) -> str:
    """Key for a budget in result dicts: 0.01 -> "b01", 0.10 -> "b10"."""
    return f"b{int(round(b * 100)):02d}"


def _check_not_constant(score: np.ndarray) -> None:
    vals = score[~np.isnan(score)]
    if vals.size == 0 or vals.min() == vals.max():
        raise ValueError("constant score")


def _sort_key(score: np.ndarray) -> np.ndarray:
    """Negated score with missing values moved to +inf, so ascending sort puts them last."""
    neg = -score
    neg[np.isnan(neg)] = np.inf
    return neg


def rank_order(score, tie_key) -> np.ndarray:
    """Row indices in ranked order: score descending, NaN last, tie_key ascending.

    Element 0 is the index of the top-ranked row. Raises ValueError("constant score")
    if all non-missing scores are equal or every score is missing.
    """
    s = np.asarray(score, dtype=float)
    _check_not_constant(s)
    return np.lexsort((np.asarray(tie_key, dtype=np.int64), _sort_key(s)))


def year_sorted_order(fyear, score, tie_key) -> np.ndarray:
    """Row indices sorted by fiscal year, then the ranking order within each year."""
    s = np.asarray(score, dtype=float)
    _check_not_constant(s)
    t = np.asarray(tie_key, dtype=np.int64)
    fy = np.asarray(fyear, dtype=np.int64)
    # Same result as np.lexsort((t, key, fy)), done as staged sorts because it is faster.
    o = np.argsort(t)
    ts = t[o]
    if (ts[1:] == ts[:-1]).any():
        o = np.argsort(t, kind="stable")
    o = o[np.argsort(_sort_key(s)[o], kind="stable")]
    f = fy[o] - fy.min()
    if f.max() < np.iinfo(np.int16).max:
        f = f.astype(np.int16)
    return o[np.argsort(f, kind="stable")]


def flag_by_budget(fyear, score, tie_key, b: float) -> np.ndarray:
    """Boolean flags: the top ceil(b * N_y) rows of each fiscal year y."""
    fy = np.asarray(fyear, dtype=np.int64)
    order = year_sorted_order(fy, score, tie_key)
    fy_sorted = fy[order]
    start = np.flatnonzero(np.r_[True, fy_sorted[1:] != fy_sorted[:-1]])
    counts = np.diff(np.r_[start, fy.size])
    inv = np.repeat(np.arange(start.size), counts)
    k = np.ceil(np.round(b * counts, 9)).astype(np.int64)
    pos_in_year = np.arange(fy.size) - start[inv]
    flags = np.zeros(fy.size, dtype=bool)
    flags[order] = pos_in_year < k[inv]
    return flags


def budget_metrics(preds: pd.DataFrame, score_col: str, b: float) -> dict:
    """K, TP, FP, P, N, precision, recall, prevalence and lift at yearly budget b."""
    flags = flag_by_budget(preds["fyear"], preds[score_col], preds["tie_key"], b)
    y = preds["y"].to_numpy().astype(bool)
    k, tp, p, n = int(flags.sum()), int((flags & y).sum()), int(y.sum()), int(y.size)
    precision = tp / k if k else float("nan")
    prevalence = p / n if n else float("nan")
    return {
        "budget": float(b),
        "K": k,
        "TP": tp,
        "FP": k - tp,
        "P": p,
        "N": n,
        "precision": float(precision),
        "recall": float(tp / p) if p else float("nan"),
        "prevalence": float(prevalence),
        "lift": float(precision / prevalence) if prevalence else float("nan"),
    }


def group_recall(preds: pd.DataFrame, flags, mask) -> dict:
    """Recall within a group of positives (y == 1 and mask). Flags come from all rows."""
    pos = preds["y"].to_numpy().astype(bool) & np.asarray(mask, dtype=bool)
    caught = int((pos & np.asarray(flags, dtype=bool)).sum())
    n_pos = int(pos.sum())
    return {"positives": n_pos, "caught": caught, "recall": caught / n_pos if n_pos else float("nan")}


def case_recall(preds: pd.DataFrame, flags) -> dict:
    """Case-level recall: a case is caught if at least one of its positive rows is flagged."""
    pos = preds["y"].to_numpy().astype(bool)
    caught_by_case = (
        pd.Series(np.asarray(flags, dtype=bool)[pos])
        .groupby(preds["case_id"].to_numpy()[pos])
        .any()
    )
    n_cases = int(caught_by_case.size)
    caught = int(caught_by_case.sum())
    return {"cases": n_cases, "caught": caught, "recall": caught / n_cases if n_cases else float("nan")}


def auc_metrics(y, score) -> dict:
    """PR-AUC (average precision), ROC-AUC and prevalence. Missing scores rank lowest."""
    yv = np.asarray(y, dtype=int)
    s = np.asarray(score, dtype=float).copy()
    nan = np.isnan(s)
    if nan.any():
        s[nan] = (np.nanmin(s) - 1.0) if (~nan).any() else 0.0
    two_classes = 0 < yv.sum() < yv.size
    return {
        "pr_auc": float(average_precision_score(yv, s)) if two_classes else float("nan"),
        "roc_auc": float(roc_auc_score(yv, s)) if two_classes else float("nan"),
        "prevalence": float(yv.mean()) if yv.size else float("nan"),
    }


def recall_curve(preds: pd.DataFrame, score_col: str, budgets=CURVE_BUDGETS) -> pd.DataFrame:
    """Recall and precision at each yearly budget (default 1% to 20% in 1% steps)."""
    rows = []
    for b in budgets:
        m = budget_metrics(preds, score_col, float(b))
        rows.append({k: m[k] for k in ("budget", "K", "TP", "recall", "precision")})
    return pd.DataFrame(rows, columns=["budget", "K", "TP", "recall", "precision"])


def random_expected(preds: pd.DataFrame, b: float, n_draws: int = 10_000, seed: int = 0) -> dict:
    """Random review: expected catches sum_y K_y * P_y / N_y, with a 95% range.

    The range is the 2.5th and 97.5th percentile of the sum over years of
    n_draws hypergeometric draws per year. It is labeled "expected".
    """
    rng = np.random.default_rng(seed)
    g = preds.groupby("fyear", sort=True)["y"].agg(["size", "sum"])
    total = np.zeros(n_draws, dtype=np.int64)
    expected = 0.0
    for n_y, p_y in zip(g["size"].astype(int), g["sum"].astype(int)):
        k_y = budget_k(n_y, b)
        expected += k_y * p_y / n_y
        total += rng.hypergeometric(p_y, n_y - p_y, k_y, size=n_draws)
    lo, hi = np.percentile(total, [2.5, 97.5])
    p = int(g["sum"].sum())
    return {
        "expected_TP": float(expected),
        "lo": float(lo),
        "hi": float(hi),
        "expected_recall": expected / p if p else float("nan"),
        "recall_lo": float(lo) / p if p else float("nan"),
        "recall_hi": float(hi) / p if p else float("nan"),
        "label": "expected",
    }


def flag_nothing(preds: pd.DataFrame) -> dict:
    """Accuracy of flagging no filing at all. It has zero catches."""
    n = int(len(preds))
    p = int(preds["y"].sum())
    return {"accuracy": (n - p) / n if n else float("nan"), "TP": 0, "N": n, "P": p}


def evaluate_method(preds: pd.DataFrame, score_col: str, budgets=DEFAULT_BUDGETS) -> dict:
    """Budget metrics keyed "b01", "b05", "b10", plus pr_auc, roc_auc and prevalence."""
    out: dict = {budget_key(b): budget_metrics(preds, score_col, b) for b in budgets}
    out.update(auc_metrics(preds["y"], preds[score_col]))
    return out


def period_half(fyear) -> np.ndarray:
    """Test-window halves: "2003-2005", "2006-2008", "" for any other year."""
    fy = np.asarray(fyear, dtype=np.int64)
    return np.select(
        [(fy >= 2003) & (fy <= 2005), (fy >= 2006) & (fy <= 2008)],
        ["2003-2005", "2006-2008"],
        default="",
    ).astype(object)


def size_thirds(preds: pd.DataFrame) -> pd.Series:
    """Within-year total-assets terciles: "small", "mid", "large" ("" if at is missing).

    Assets are ranked within each year (ties by tie_key), then cut into 3 equal bins.
    """
    assert_unlocked(preds)
    out = pd.Series("", index=preds.index, dtype=object)
    ok = preds["at"].notna()
    d = preds.loc[ok, ["fyear", "at", "tie_key"]].sort_values(["fyear", "at", "tie_key"])
    rank = d.groupby("fyear").cumcount()
    labels = rank.groupby(d["fyear"]).transform(
        lambda r: pd.qcut(r, 3, labels=["small", "mid", "large"]).astype(object)
    )
    out.loc[labels.index] = labels.astype(object)
    return out


def subgroup_masks(preds: pd.DataFrame, extra: dict | None = None) -> dict[str, np.ndarray]:
    """Boolean row masks for the reported subgroups. Empty groups are left out.

    Always: all, case_seen, case_unseen, period_2003-2005, period_2006-2008.
    size_small/mid/large when an "at" column is present. Any masks in extra
    (for example seen versus unseen companies) are added as given.
    """
    masks: dict[str, np.ndarray] = {"all": np.ones(len(preds), dtype=bool)}
    grp = preds["case_group"].astype(str).to_numpy()
    masks["case_seen"] = grp == "seen"
    masks["case_unseen"] = grp == "unseen"
    half = period_half(preds["fyear"])
    for lab in ("2003-2005", "2006-2008"):
        masks[f"period_{lab}"] = half == lab
    if "at" in preds.columns:
        thirds = size_thirds(preds).to_numpy()
        for lab in ("small", "mid", "large"):
            masks[f"size_{lab}"] = thirds == lab
    for name, m in (extra or {}).items():
        masks[name] = np.asarray(m, dtype=bool)
    return {k: v for k, v in masks.items() if v.any()}
