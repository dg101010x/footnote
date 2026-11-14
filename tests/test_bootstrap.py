import json

import numpy as np
import pandas as pd
import pytest

from footnote import bootstrap as bs
from footnote.evaluate import flag_by_budget


def _synthetic(n_comp=120, seed=0, p_pos=0.15):
    """Synthetic preds: companies with 1 to 4 years, some positives in seen/unseen cases."""
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(n_comp):
        years = sorted(rng.choice(np.arange(2003, 2009), size=rng.integers(1, 5), replace=False))
        positive = rng.random() < p_pos
        group = "seen" if rng.random() < 0.4 else "unseen"
        for yr in years:
            yy = int(positive and rng.random() < 0.7)
            rows.append(
                {
                    "gvkey": 1000 + c,
                    "fyear": yr,
                    "y": yy,
                    "case_id": float(c) if yy else np.nan,
                    "case_group": group if yy else "",
                }
            )
    df = pd.DataFrame(rows)
    n = len(df)
    df["tie_key"] = rng.permutation(n)
    df["oracle"] = df["y"] + 0.01 * rng.random(n)
    df["random"] = rng.random(n)
    df["random2"] = rng.random(n)
    df["tied"] = np.round(rng.random(n), 1)
    df.loc[rng.random(n) < 0.1, "tied"] = np.nan
    return df


def test_verdict_helper():
    assert bs.verdict(0.01, 0.2) == "beats"
    assert bs.verdict(-0.2, -0.01) == "loses to"
    assert bs.verdict(-0.1, 0.1) == "cannot tell apart"
    assert bs.verdict(0.0, 0.0) == "cannot tell apart"
    assert bs.verdict(0.0, 0.3) == "cannot tell apart"


def test_deterministic_with_seed_and_json_serializable():
    df = _synthetic()
    kw = dict(methods=["oracle", "random"], model="oracle", rule="random", logistic="random2", n_boot=50)
    a = bs.paired_bootstrap(df, seed=0, **kw)
    b = bs.paired_bootstrap(df, seed=0, **kw)
    c = bs.paired_bootstrap(df, seed=1, **kw)
    assert json.dumps(a, allow_nan=False) == json.dumps(b, allow_nan=False)
    assert a != c
    assert a["settings"]["unit"] == "company" and a["settings"]["n_boot"] == 50


def test_identical_methods_give_zero_difference():
    df = _synthetic()
    df["copy_of_random"] = df["random"]
    r = bs.paired_bootstrap(
        df, ["random"], model="random", rule="copy_of_random", logistic="random", n_boot=50
    )["stats"]
    for name in ("model_minus_rule_all", "model_minus_rule_seen", "model_minus_rule_unseen",
                 "model_minus_logistic_all", "gap_diff", "case_model_minus_rule"):
        s = r[name]
        assert s["point"] == 0.0 and s["lo"] == 0.0 and s["hi"] == 0.0
        assert s["verdict"] == "cannot tell apart"
        assert 0 < s["n_valid"] <= 50


def test_oracle_beats_random():
    df = _synthetic(n_comp=300, p_pos=0.05)
    r = bs.paired_bootstrap(
        df, ["oracle", "random"], model="oracle", rule="random", logistic="random2", n_boot=50
    )["stats"]
    assert r["model_minus_rule_all"]["verdict"] == "beats"
    assert r["model_minus_logistic_all"]["verdict"] == "beats"
    assert r["recall_all.oracle"]["point"] == 1.0
    assert r["recall_all.random"]["verdict"] is None


def test_point_estimates_match_real_flags():
    df = _synthetic()
    r = bs.paired_bootstrap(df, ["tied"], model="tied", rule="random", logistic="oracle", n_boot=5)["stats"]
    flags = flag_by_budget(df["fyear"], df["tied"], df["tie_key"], 0.10)
    y = df["y"].to_numpy().astype(bool)
    assert r["recall_all.tied"]["point"] == pytest.approx((flags & y).sum() / y.sum())
    seen = y & (df["case_group"] == "seen").to_numpy()
    assert r["recall_seen.tied"]["point"] == pytest.approx((flags & seen).sum() / seen.sum())


def test_replicate_copies_get_distinct_tie_keys():
    df = _synthetic(n_comp=10)
    companies = np.unique(df["gvkey"])
    rep = bs.replicate_frame(df, np.array([0, 0, 3]), companies)
    assert rep["tie_key"].is_unique
    n0 = (df["gvkey"] == companies[0]).sum()
    assert len(rep) == 2 * n0 + (df["gvkey"] == companies[3]).sum()
    first = rep.loc[rep["copy"] == 0, "tie_key"].to_numpy()
    second = rep.loc[rep["copy"] == 1, "tie_key"].to_numpy()
    assert (second - first == 1).all()


def test_fast_path_matches_explicit_replicate():
    """Bootstrap counts equal flag_by_budget on explicitly built replicates."""
    df = _synthetic(n_comp=80, seed=4)
    companies = np.unique(df["gvkey"])
    n_boot = 8
    res = bs.paired_bootstrap(
        df, ["tied"], model="tied", rule="random", logistic="oracle", n_boot=n_boot, seed=7
    )["stats"]
    rng = np.random.default_rng(7)
    expected_all, expected_case = [], []
    for _ in range(n_boot):
        draws = rng.integers(0, companies.size, size=companies.size)
        rep = bs.replicate_frame(df, draws, companies)
        flags = flag_by_budget(rep["fyear"], rep["tied"], rep["tie_key"], 0.10)
        y = rep["y"].to_numpy().astype(bool)
        if y.sum():
            expected_all.append((flags & y).sum() / y.sum())
        pos = rep.loc[y].assign(f=flags[y])
        units = pos.groupby(["case_id", "gvkey", "copy"])["f"].any()
        if len(units):
            expected_case.append(units.mean())
    lo, hi = np.percentile(expected_all, [2.5, 97.5])
    assert res["recall_all.tied"]["lo"] == pytest.approx(lo)
    assert res["recall_all.tied"]["hi"] == pytest.approx(hi)
    clo, chi = np.percentile(expected_case, [2.5, 97.5])
    assert res["case_recall.tied"]["lo"] == pytest.approx(clo)
    assert res["case_recall.tied"]["hi"] == pytest.approx(chi)


def test_invalid_replicates_excluded():
    df = _synthetic(n_comp=300, seed=2)
    # Keep a single unseen positive company so many replicates have no unseen positives.
    unseen_comp = df.loc[df["case_group"] == "unseen", "gvkey"].unique()
    drop = df["gvkey"].isin(unseen_comp[1:])
    df = df.loc[~drop].reset_index(drop=True)
    r = bs.paired_bootstrap(df, ["random"], model="oracle", rule="random", logistic="random2", n_boot=50)
    s = r["stats"]
    assert s["recall_unseen.oracle"]["n_valid"] < 50
    assert s["recall_all.oracle"]["n_valid"] == 50
    assert s["gap_diff"]["n_valid"] == s["recall_unseen.oracle"]["n_valid"]


def test_missing_method_column_raises():
    with pytest.raises(KeyError):
        bs.paired_bootstrap(_synthetic(), ["nope"], model="oracle", rule="random", logistic="random2", n_boot=2)
