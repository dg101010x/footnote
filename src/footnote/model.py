"""LightGBM primary model: four candidates, selection rule, five-seed final model (spec section 7).

Scores are raw LightGBM margins used as a ranking, not probabilities.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import lightgbm
import numpy as np
import pandas as pd

from .features import FEATURE_NAMES

MODEL_TYPE = "lightgbm"
FIXED_PARAMS = dict(
    learning_rate=0.03,
    min_child_samples=50,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    verbose=-1,
)
CANDIDATES = {
    "G1": {"num_leaves": 7, "n_estimators": 150},
    "G2": {"num_leaves": 7, "n_estimators": 300},
    "G3": {"num_leaves": 15, "n_estimators": 150},
    "G4": {"num_leaves": 15, "n_estimators": 300},
}
FINAL_SEEDS = (0, 1, 2, 3, 4)


def scale_pos_weight(y) -> float:
    """Negatives over positives in the fit rows."""
    y = np.asarray(y).astype(int)
    pos = int(y.sum())
    if pos == 0:
        raise ValueError("No positive rows in the fit rows")
    return float((len(y) - pos) / pos)


def _check_columns(X: pd.DataFrame) -> None:
    assert list(X.columns) == FEATURE_NAMES, "X columns must equal features.FEATURE_NAMES in order"


def candidate_params(candidate: str, y, seed: int = 0) -> dict:
    """Full LightGBM parameters for one candidate on these fit labels."""
    return {**FIXED_PARAMS, **CANDIDATES[candidate], "scale_pos_weight": scale_pos_weight(y), "random_state": seed}


def fit_one(X: pd.DataFrame, y, candidate: str, seed: int = 0) -> tuple[lightgbm.LGBMClassifier, float]:
    """Fit one candidate; return the model and the fit runtime in seconds."""
    _check_columns(X)
    clf = lightgbm.LGBMClassifier(**candidate_params(candidate, y, seed))
    t0 = time.perf_counter()
    clf.fit(X, np.asarray(y).astype(int))
    return clf, time.perf_counter() - t0


def raw_score(model: lightgbm.LGBMClassifier, X: pd.DataFrame) -> np.ndarray:
    """Raw margin scores (a ranking)."""
    _check_columns(X)
    return np.asarray(model.predict(X, raw_score=True), dtype="float64")


def select_candidate(rows: list[dict]) -> tuple[str, str]:
    """Pick the highest recall_b10, then average_precision, then the smaller model."""
    def key(r: dict):
        size = r["num_leaves"] * r["n_estimators"]
        return (-r["recall_b10"], -r["average_precision"], size, r["num_leaves"])

    ranked = sorted(rows, key=key)
    best = ranked[0]
    if len(ranked) == 1:
        how = "it was the only candidate"
    elif ranked[1]["recall_b10"] != best["recall_b10"]:
        how = "it had the highest validation recall at the 10% yearly budget"
    elif ranked[1]["average_precision"] != best["average_precision"]:
        how = "it tied on validation recall at the 10% yearly budget and had the higher average precision"
    else:
        how = "it tied on validation recall at the 10% yearly budget and on average precision, and is the smaller model"
    reason = (
        f"{best['candidate']} was picked because {how} "
        f"(recall {best['recall_b10']:.4f}, average precision {best['average_precision']:.4f})."
    )
    return best["candidate"], reason


class FinalModel:
    """The chosen candidate refit with seeds 0 to 4; the score is the mean raw score."""

    def __init__(self, candidate: str, seeds: tuple[int, ...] = FINAL_SEEDS) -> None:
        if candidate not in CANDIDATES:
            raise KeyError(f"Unknown candidate {candidate!r}")
        self.candidate = candidate
        self.seeds = tuple(int(s) for s in seeds)
        self.boosters: list[lightgbm.Booster] = []
        self.params: dict | None = None

    def fit(self, X: pd.DataFrame, y) -> "FinalModel":
        _check_columns(X)
        self.boosters = []
        for s in self.seeds:
            clf, _ = fit_one(X, y, self.candidate, seed=s)
            self.boosters.append(clf.booster_)
        self.params = {k: v for k, v in candidate_params(self.candidate, y).items() if k != "random_state"}
        return self

    def _x(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.boosters:
            raise RuntimeError("FinalModel is not fitted")
        _check_columns(X)
        return X

    def score(self, X: pd.DataFrame) -> np.ndarray:
        x = self._x(X)
        return np.mean([b.predict(x, raw_score=True) for b in self.boosters], axis=0)

    def contrib(self, X: pd.DataFrame) -> pd.DataFrame:
        """Mean per-feature contributions to the raw score; the row sum including bias equals score()."""
        x = self._x(X)
        arr = np.mean([b.predict(x, pred_contrib=True) for b in self.boosters], axis=0)
        return pd.DataFrame(arr, index=X.index, columns=FEATURE_NAMES + ["bias"])

    def save(self, dir: Path | str) -> Path:
        d = Path(dir)
        d.mkdir(parents=True, exist_ok=True)
        if not self.boosters:
            raise RuntimeError("FinalModel is not fitted")
        for s, b in zip(self.seeds, self.boosters):
            b.save_model(str(d / f"seed_{s}.txt"))
        meta = {
            "model_type": MODEL_TYPE,
            "candidate": self.candidate,
            "params": self.params,
            "seeds": list(self.seeds),
            "feature_names": FEATURE_NAMES,
            "lightgbm_version": lightgbm.__version__,
        }
        (d / "model_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
        return d

    @classmethod
    def load(cls, dir: Path | str) -> "FinalModel":
        d = Path(dir)
        meta = json.loads((d / "model_meta.json").read_text())
        if meta["feature_names"] != FEATURE_NAMES:
            raise ValueError("Saved feature names do not match features.FEATURE_NAMES")
        fm = cls(meta["candidate"], tuple(meta["seeds"]))
        fm.params = meta["params"]
        fm.boosters = [lightgbm.Booster(model_file=str(d / f"seed_{s}.txt")) for s in fm.seeds]
        return fm
