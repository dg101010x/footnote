import json

import numpy as np
import pandas as pd
import pytest

from footnote import model
from footnote.features import FEATURE_NAMES


@pytest.fixture(scope="module")
def xy():
    rng = np.random.default_rng(0)
    n = 2000
    X = pd.DataFrame(rng.normal(size=(n, len(FEATURE_NAMES))), columns=FEATURE_NAMES)
    X.iloc[::11, 5] = np.nan
    logit = 1.5 * X.iloc[:, 0] - X.iloc[:, 3] + rng.normal(0, 1, n) - 2.5
    y = (logit > 0).astype(int).to_numpy()
    return X, y


def test_constants():
    assert model.MODEL_TYPE == "lightgbm"
    assert set(model.CANDIDATES) == {"G1", "G2", "G3", "G4"}
    assert model.CANDIDATES["G3"] == {"num_leaves": 15, "n_estimators": 150}
    assert model.scale_pos_weight([0, 0, 0, 1]) == 3.0


def test_fit_one_asserts_column_order(xy):
    X, y = xy
    with pytest.raises(AssertionError):
        model.fit_one(X[FEATURE_NAMES[::-1]], y, "G1")
    clf, secs = model.fit_one(X, y, "G1", seed=0)
    assert secs > 0
    p = clf.get_params()
    assert p["num_leaves"] == 7 and p["n_estimators"] == 150 and p["random_state"] == 0
    assert p["scale_pos_weight"] == pytest.approx((y == 0).sum() / y.sum())
    s = model.raw_score(clf, X)
    assert s.shape == (len(X),) and np.isfinite(s).all()
    np.testing.assert_array_equal(s, model.raw_score(model.fit_one(X, y, "G1", seed=0)[0], X))


def _row(c, r, ap, leaves, trees):
    return {"candidate": c, "recall_b10": r, "average_precision": ap, "num_leaves": leaves, "n_estimators": trees}


def test_select_candidate_tie_breaks():
    assert model.select_candidate([_row("G1", 0.2, 0.1, 7, 150), _row("G4", 0.3, 0.05, 15, 300)])[0] == "G4"
    name, reason = model.select_candidate([_row("G1", 0.3, 0.1, 7, 150), _row("G4", 0.3, 0.2, 15, 300)])
    assert name == "G4" and "average precision" in reason
    rows = [_row(c, 0.3, 0.2, l, t) for c, (l, t) in
            [("G4", (15, 300)), ("G2", (7, 300)), ("G3", (15, 150)), ("G1", (7, 150))]]
    name, reason = model.select_candidate(rows)
    assert name == "G1" and "smaller" in reason and reason.endswith(".")
    # equal size (7*300 vs hypothetical 15*140) -> fewer leaves
    assert model.select_candidate([_row("A", 0.3, 0.2, 15, 140), _row("B", 0.3, 0.2, 7, 300)])[0] == "B"


def test_final_model_save_load_and_contrib(xy, tmp_path):
    X, y = xy
    fm = model.FinalModel("G1").fit(X, y)
    assert len(fm.boosters) == 5
    s = fm.score(X)
    c = fm.contrib(X)
    assert list(c.columns) == FEATURE_NAMES + ["bias"]
    np.testing.assert_allclose(c.sum(axis=1).to_numpy(), s, atol=1e-9)
    fm.save(tmp_path)
    meta = json.loads((tmp_path / "model_meta.json").read_text())
    assert meta["candidate"] == "G1" and meta["seeds"] == [0, 1, 2, 3, 4]
    assert meta["model_type"] == "lightgbm" and meta["feature_names"] == FEATURE_NAMES
    assert all((tmp_path / f"seed_{k}.txt").exists() for k in range(5))
    loaded = model.FinalModel.load(tmp_path)
    np.testing.assert_allclose(loaded.score(X), s, atol=1e-9)
    np.testing.assert_allclose(loaded.contrib(X).to_numpy(), c.to_numpy(), atol=1e-9)
    single = model.fit_one(X, y, "G1", seed=0)[0]
    assert not np.allclose(model.raw_score(single, X), s)  # seed average differs from one seed
