import numpy as np
import pandas as pd
import pytest

from footnote import baselines, data, features
from footnote.splits import LockedSplitError


def _raw(n: int, fyear, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({c: rng.uniform(1, 100, n) for c in data.RAW_ITEMS})
    df["at"] = 1000.0
    df["fyear"] = fyear
    return df


def _two_years() -> pd.DataFrame:
    df = _raw(6, [1995, 1995, 1995, 1996, 1996, 1996])
    df["sale"] = 100.0
    df["rect"] = [10.0, 30.0, 20.0, 5.0, 50.0, 1.0]
    df["ppegt"] = [100.0, 500.0, 300.0, 0.0, 0.0, 0.0]
    df["che"] = [0.0, 0.0, 0.0, 100.0, 900.0, 500.0]
    df.loc[2, "sale"] = 0.0  # rule 2a undefined
    df.loc[5, "at"] = 0.0  # rule 2b undefined
    return df


def test_rule_direction_and_missing():
    s = baselines.rule_scores(_two_years())
    assert list(s.columns) == list(baselines.RULES)
    np.testing.assert_allclose(s["rule_2a"].iloc[[0, 1, 3, 4]], [0.1, 0.3, 0.05, 0.5])
    assert np.isnan(s.loc[2, "rule_2a"])
    # soft assets share: higher means less plant and cash
    np.testing.assert_allclose(s["rule_2b"].iloc[:5], [0.9, 0.5, 0.7, 0.9, 0.1])
    assert np.isnan(s.loc[5, "rule_2b"])


def test_rule_2c_percentiles_within_year():
    s = baselines.rule_scores(_two_years())
    # 1995: 2a pct among non-missing {0.1, 0.3} -> 0.5, 1.0, missing 0; 2b {0.9,0.5,0.7} -> 1, 1/3, 2/3
    # 1996: 2a {0.05, 0.5, 0.01} -> 2/3, 1, 1/3; 2b {0.9, 0.1, NaN} -> 1, 0.5, 0
    expected = [(0.5 + 1) / 2, (1 + 1 / 3) / 2, (0 + 2 / 3) / 2, (2 / 3 + 1) / 2, (1 + 0.5) / 2, (1 / 3 + 0) / 2]
    np.testing.assert_allclose(s["rule_2c"].to_numpy(), expected)
    assert s.index.equals(_two_years().index)


def test_rule_2c_years_not_pooled():
    a = _two_years()
    b = a.copy()
    b.loc[b["fyear"] == 1996, ["rect"]] *= 7  # changes 1996 values but not their order
    pd.testing.assert_series_equal(
        baselines.rule_scores(a)["rule_2c"], baselines.rule_scores(b)["rule_2c"]
    )


def test_rules_refuse_locked_years():
    with pytest.raises(LockedSplitError):
        baselines.rule_scores(_raw(3, 2005))


def _fit_frame(n: int = 400, seed: int = 1) -> tuple[pd.DataFrame, np.ndarray]:
    raw = _raw(n, 1995, seed)
    raw["at"] = np.random.default_rng(seed).uniform(50, 5000, n)
    X = features.build_features(raw)
    X.iloc[::7, 3] = np.nan
    X.iloc[0, 10] = 1e9  # extreme value to be clipped
    y = (X["rect_at"].fillna(0) + np.random.default_rng(seed).normal(0, 0.02, n) > 0.06).astype(int)
    return X, y.to_numpy()


def test_logistic_stats_from_fit_rows_only():
    X, y = _fit_frame()
    m = baselines.LogisticBaseline().fit(X, y)
    x = X[features.LOGIT_FEATURES]
    pd.testing.assert_series_equal(m.lo_, x.quantile(0.01))
    pd.testing.assert_series_equal(m.median_, x.median())
    filled = x.clip(lower=m.lo_, upper=m.hi_, axis=1).fillna(m.median_)
    np.testing.assert_allclose(m.mean_, filled.mean())
    before = (m.lo_.copy(), m.hi_.copy(), m.median_.copy(), m.mean_.copy(), m.std_.copy(), m.clf_.coef_.copy())
    other, _ = _fit_frame(200, seed=9)
    other.iloc[:, :] = other.to_numpy() * 50
    s = m.score(other)
    assert np.isfinite(s).all() and s.shape == (200,)
    after = (m.lo_, m.hi_, m.median_, m.mean_, m.std_, m.clf_.coef_)
    for a, b in zip(before, after):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_logistic_nan_constant_and_deterministic():
    X, y = _fit_frame()
    X["log_assets"] = 3.0  # constant feature: std 0 becomes 1
    s1 = baselines.LogisticBaseline().fit(X, y).score(X)
    s2 = baselines.LogisticBaseline().fit(X, y).score(X)
    assert np.isfinite(s1).all()
    np.testing.assert_array_equal(s1, s2)
    allnan = X.copy()
    allnan.iloc[:5, :] = np.nan
    assert np.isfinite(baselines.LogisticBaseline().fit(X, y).score(allnan)).all()


def test_logistic_selects_columns_from_any_frame():
    X, y = _fit_frame()
    shuffled = X[X.columns[::-1]].assign(extra=1.0)
    m = baselines.LogisticBaseline().fit(X, y)
    np.testing.assert_allclose(m.score(shuffled), m.score(X))
    ref = baselines.extra_reference()
    assert ref.features == data.READY_RATIOS and ref.label == "extra-input reference"
    rr = pd.DataFrame(np.random.default_rng(0).normal(size=(len(y), 14)), columns=data.READY_RATIOS)
    rr["fyear"] = 1995
    assert ref.fit(rr, y).score(rr).shape == (len(y),)
