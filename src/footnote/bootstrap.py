"""Paired company bootstrap for recall at a yearly budget (build spec section 8).

Each replicate draws len(companies) company ids with replacement. A company drawn
m times contributes m copies of all its rows. Copy tie keys are
original tie_key * 1_000_003 + draw_position, so copies of one row sit next to
each other in the ranking and never change the order of other rows. Because of
that, the replicate ranking inside a year is the original ranking with each row
repeated m times. The code uses this to count flagged copies without re-sorting:
in year order, a row whose copies occupy ranks [c - m, c) has
clip(K_y - (c - m), 0, m) flagged copies. This gives exactly the flags that
flag_by_budget returns on the explicitly built replicate (checked in tests).

Case-level recall in a replicate counts one case unit per (case_id, company, copy).
Copy j of a company is caught if some positive row of that case in that copy is
flagged. A case that spans two companies therefore counts as two units in a
replicate (there are 3 such cases in the data), while the point estimate on the
real data counts it once.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .evaluate import case_recall, flag_by_budget, year_sorted_order

COPY_MULTIPLIER = 1_000_003
VERDICTS = ("beats", "cannot tell apart", "loses to")


def verdict(lo: float, hi: float) -> str:
    """"beats" if lo > 0, "loses to" if hi < 0, else "cannot tell apart"."""
    if lo > 0:
        return "beats"
    if hi < 0:
        return "loses to"
    return "cannot tell apart"


def replicate_frame(preds: pd.DataFrame, draws: np.ndarray, companies: np.ndarray) -> pd.DataFrame:
    """Explicit replicate: all rows of each drawn company, copy tie keys made distinct.

    draws holds indexes into companies. Used by tests to check the fast path.
    """
    parts = []
    for pos, ci in enumerate(draws):
        rows = preds.loc[preds["gvkey"] == companies[ci]].copy()
        rows["tie_key"] = rows["tie_key"].astype(np.int64) * COPY_MULTIPLIER + pos
        rows["copy"] = pos
        parts.append(rows)
    return pd.concat(parts, ignore_index=True)


def _interval(values: np.ndarray, valid: np.ndarray) -> tuple[float | None, float | None, int]:
    v = values[valid]
    if v.size == 0:
        return None, None, 0
    lo, hi = np.percentile(v, [2.5, 97.5])
    return float(lo), float(hi), int(v.size)


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b > 0 else float("nan")


def paired_bootstrap(
    preds: pd.DataFrame,
    methods: list[str],
    model: str = "model",
    rule: str = "best_rule",
    logistic: str = "logistic",
    b: float = 0.10,
    n_boot: int = 1000,
    seed: int = 0,
) -> dict:
    """Paired company bootstrap of recall at yearly budget b for every method.

    Returns {"settings": {...}, "stats": {name: {point, lo, hi, n_valid, verdict}}}.
    Recall stats are "recall_{all,seen,unseen}.<method>" and "case_recall.<method>"
    for model and rule. Difference stats (with a verdict) are
    model_minus_rule_*, model_minus_logistic_*, gap_model, gap_rule, gap_diff
    and case_model_minus_rule. Points come from the real data; intervals are the
    2.5th and 97.5th percentiles over replicates whose denominators are above zero.
    """
    all_methods = list(dict.fromkeys([*methods, model, rule, logistic]))
    missing = [m for m in all_methods if m not in preds.columns]
    if missing:
        raise KeyError(f"score columns missing from preds: {missing}")

    df = preds.reset_index(drop=True)
    n = len(df)
    fyear = df["fyear"].to_numpy(dtype=np.int64)
    tie = df["tie_key"].to_numpy(dtype=np.int64)
    y = df["y"].to_numpy().astype(bool)
    grp = df["case_group"].astype(str).to_numpy()
    seen = y & (grp == "seen")
    unseen = y & (grp == "unseen")

    companies, row_comp = np.unique(df["gvkey"].to_numpy(), return_inverse=True)
    n_comp = companies.size
    years, year_idx = np.unique(fyear, return_inverse=True)
    n_years = years.size

    # Ranking per method: rows sorted by year, then score order. Year blocks are
    # identical across methods, so the year of each sorted position is shared.
    orders = {m: year_sorted_order(fyear, df[m].to_numpy(dtype=float), tie) for m in all_methods}
    year_of_sorted = np.sort(year_idx)

    # Case units: (case_id, company) groups of positive rows, for model and rule.
    pos_idx = np.flatnonzero(y)
    case_key = pd.MultiIndex.from_arrays([df["case_id"].to_numpy()[pos_idx], row_comp[pos_idx]])
    unit_codes, _ = pd.factorize(case_key)
    unit_order = np.argsort(unit_codes, kind="stable")
    unit_rows = pos_idx[unit_order]
    unit_starts = np.flatnonzero(np.r_[True, np.diff(unit_codes[unit_order]) != 0])
    unit_comp = row_comp[unit_rows[unit_starts]]

    def flagged_copies(m: str, mult: np.ndarray, k_year: np.ndarray, year_start: np.ndarray) -> np.ndarray:
        order = orders[m]
        ms = mult[order]
        before = np.cumsum(ms) - ms - year_start[year_of_sorted]
        f = np.clip(k_year[year_of_sorted] - before, 0, ms)
        out = np.empty(n, dtype=np.int64)
        out[order] = f
        return out

    def one_pass(mult_comp: np.ndarray) -> dict:
        mult = mult_comp[row_comp]
        n_y = np.bincount(year_idx, weights=mult, minlength=n_years).astype(np.int64)
        k_year = np.ceil(np.round(b * n_y, 9)).astype(np.int64)
        year_start = np.r_[0, np.cumsum(n_y)[:-1]]
        res = {
            "P_all": int(mult[y].sum()),
            "P_seen": int(mult[seen].sum()),
            "P_unseen": int(mult[unseen].sum()),
            "cases": int(mult_comp[unit_comp].sum()),
        }
        for m in all_methods:
            f = flagged_copies(m, mult, k_year, year_start)
            res[("all", m)] = int(f[y].sum())
            res[("seen", m)] = int(f[seen].sum())
            res[("unseen", m)] = int(f[unseen].sum())
            if m in (model, rule):
                res[("case", m)] = int(np.maximum.reduceat(f[unit_rows], unit_starts).sum()) if unit_rows.size else 0
        return res

    def stat_values(r: dict) -> dict[str, tuple[float, bool]]:
        """Each stat's value and whether its denominators are above zero."""
        ok = {"all": r["P_all"] > 0, "seen": r["P_seen"] > 0, "unseen": r["P_unseen"] > 0}
        den = {"all": r["P_all"], "seen": r["P_seen"], "unseen": r["P_unseen"]}
        rec = {(g, m): _safe_div(r[(g, m)], den[g]) for g in den for m in all_methods}
        out: dict[str, tuple[float, bool]] = {}
        for m in all_methods:
            for g in den:
                out[f"recall_{g}.{m}"] = (rec[(g, m)], ok[g])
        for g in den:
            out[f"model_minus_rule_{g}"] = (rec[(g, model)] - rec[(g, rule)], ok[g])
            out[f"model_minus_logistic_{g}"] = (rec[(g, model)] - rec[(g, logistic)], ok[g])
        both = ok["seen"] and ok["unseen"]
        gap_m = rec[("seen", model)] - rec[("unseen", model)]
        gap_r = rec[("seen", rule)] - rec[("unseen", rule)]
        out["gap_model"] = (gap_m, both)
        out["gap_rule"] = (gap_r, both)
        out["gap_diff"] = (gap_m - gap_r, both)
        c_ok = r["cases"] > 0
        cm = _safe_div(r[("case", model)], r["cases"])
        cr = _safe_div(r[("case", rule)], r["cases"])
        out[f"case_recall.{model}"] = (cm, c_ok)
        out[f"case_recall.{rule}"] = (cr, c_ok)
        out["case_model_minus_rule"] = (cm - cr, c_ok)
        return out

    # Point estimates from the real data (every company once).
    point = {k: v for k, (v, _) in stat_values(one_pass(np.ones(n_comp, dtype=np.int64))).items()}
    for m in (model, rule):
        flags = flag_by_budget(fyear, df[m].to_numpy(dtype=float), tie, b)
        point[f"case_recall.{m}"] = case_recall(df, flags)["recall"]
    point["case_model_minus_rule"] = point[f"case_recall.{model}"] - point[f"case_recall.{rule}"]

    rng = np.random.default_rng(seed)
    names = list(point)
    vals = np.full((n_boot, len(names)), np.nan)
    valid = np.zeros((n_boot, len(names)), dtype=bool)
    for i in range(n_boot):
        draws = rng.integers(0, n_comp, size=n_comp)
        sv = stat_values(one_pass(np.bincount(draws, minlength=n_comp)))
        for j, name in enumerate(names):
            vals[i, j], valid[i, j] = sv[name]

    diff_names = {
        *(f"model_minus_rule_{g}" for g in ("all", "seen", "unseen")),
        *(f"model_minus_logistic_{g}" for g in ("all", "seen", "unseen")),
        "gap_model", "gap_rule", "gap_diff", "case_model_minus_rule",
    }
    stats: dict[str, dict] = {}
    for j, name in enumerate(names):
        lo, hi, n_valid = _interval(vals[:, j], valid[:, j])
        p = point[name]
        stats[name] = {
            "point": None if np.isnan(p) else float(p),
            "lo": lo,
            "hi": hi,
            "n_valid": n_valid,
            "verdict": verdict(lo, hi) if (name in diff_names and n_valid) else None,
        }
    return {
        "settings": {
            "n_boot": int(n_boot),
            "seed": int(seed),
            "b": float(b),
            "unit": "company",
            "methods": all_methods,
            "model": model,
            "rule": rule,
            "logistic": logistic,
            "case_unit": "(case_id, company, copy)",
        },
        "stats": stats,
    }
