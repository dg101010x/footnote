"""Fair baselines: three simple rules and a logistic regression (spec section 6).

All scores are rankings: higher means more suspicious. None is a probability.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from . import data, features
from .splits import assert_unlocked

RULES = {
    "rule_2a": "high receivables to sales (rect / sale)",
    "rule_2b": "high soft-assets share ((at - ppegt - che) / at)",
    "rule_2c": "average of the within-year percentile ranks of 2a and 2b",
}


def _year_pct_rank(values: pd.Series, years: pd.Series) -> pd.Series:
    """Percentile rank within each fiscal year among non-missing values; missing gives 0."""
    pct = values.groupby(years).rank(pct=True)
    return pct.fillna(0.0)


def rule_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Scores of rules 2a, 2b and 2c for company-year rows (fyear plus the 28 raw items).

    2a and 2b keep NaN where the ratio is undefined, so evaluation ranks them last.
    2c averages the within-year percentile ranks of 2a and 2b, with a missing
    component counted as percentile 0, so it is never NaN.
    """
    feats = features.build_features(df)  # checks the test lock
    out = pd.DataFrame(index=df.index)
    out["rule_2a"] = feats["rec_to_sales"].astype("float64")
    out["rule_2b"] = feats["soft_assets_share"].astype("float64")
    years = df["fyear"]
    out["rule_2c"] = (
        _year_pct_rank(out["rule_2a"], years) + _year_pct_rank(out["rule_2b"], years)
    ) / 2.0
    return out


class LogisticBaseline:
    """Logistic regression with fit-row preprocessing, no tuning.

    Preprocessing order, all statistics from the fit rows only:
    1. clip each feature to its 1st and 99th percentile (computed on raw fit values, NaN ignored);
    2. fill missing values with the fit-row median (computed on raw fit values, NaN ignored);
    3. standardize with the mean and standard deviation of the clipped and filled
       fit values (a standard deviation of 0 is replaced by 1).
    Columns are selected by name from whatever frame is given.
    """

    def __init__(self, features: list[str] = features.LOGIT_FEATURES, label: str = "logistic") -> None:
        self.features = list(features)
        self.label = label
        self.lo_: pd.Series | None = None
        self.hi_: pd.Series | None = None
        self.median_: pd.Series | None = None
        self.mean_: pd.Series | None = None
        self.std_: pd.Series | None = None
        self.clf_: LogisticRegression | None = None

    def _select(self, X: pd.DataFrame) -> pd.DataFrame:
        if "fyear" in X:
            assert_unlocked(X)
        return X[self.features].astype("float64")

    def _transform(self, x: pd.DataFrame) -> np.ndarray:
        x = x.clip(lower=self.lo_, upper=self.hi_, axis=1)
        x = x.fillna(self.median_)
        return ((x - self.mean_) / self.std_).to_numpy()

    def fit(self, X: pd.DataFrame, y) -> "LogisticBaseline":
        x = self._select(X)
        self.lo_ = x.quantile(0.01)
        self.hi_ = x.quantile(0.99)
        self.median_ = x.median()
        filled = x.clip(lower=self.lo_, upper=self.hi_, axis=1).fillna(self.median_)
        self.mean_ = filled.mean()
        std = filled.std(ddof=0)
        self.std_ = std.where(std > 0, 1.0)
        self.clf_ = LogisticRegression(class_weight="balanced", C=1.0, max_iter=1000)
        self.clf_.fit(self._transform(x), np.asarray(y).astype(int))
        return self

    def score(self, X: pd.DataFrame) -> np.ndarray:
        """Decision function values (a ranking, not a probability)."""
        if self.clf_ is None:
            raise RuntimeError("LogisticBaseline is not fitted")
        return self.clf_.decision_function(self._transform(self._select(X)))


EXTRA_REF_FEATURES = list(data.READY_RATIOS)


def extra_reference() -> LogisticBaseline:
    """Optional logistic regression on the 14 ready-made ratios, an "extra-input reference".

    It uses inputs the fair methods do not have (most use the prior year), so it is
    not a fair competitor. Its ratios are columns of the raw frame, not of build_features.
    """
    return LogisticBaseline(EXTRA_REF_FEATURES, label="extra-input reference")
