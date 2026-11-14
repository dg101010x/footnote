import math

import numpy as np
import pandas as pd
import pytest

from footnote import features as F
from footnote.data import RAW_ITEMS
from footnote.splits import LockedSplitError


def _row(**kw):
    base = {c: 1.0 for c in RAW_ITEMS}
    base.update(kw)
    return pd.DataFrame([base])


def test_counts():
    assert len(F.FEATURE_NAMES) == 61
    assert len(set(F.FEATURE_NAMES)) == 61
    assert len(F.SCALED_FEATURES) == 25
    assert "at_at" not in F.FEATURE_NAMES
    assert len(F.LOGIT_FEATURES) == 33


def test_known_values():
    r = _row(at=200.0, act=50.0, lct=25.0, lt=120.0, sale=100.0, cogs=60.0, rect=41.0,
             invt=10.0, ppegt=80.0, che=20.0, ceq=40.0, prcc_f=4.0, csho=20.0)
    f = F.compute_features(r).iloc[0]
    assert f["current_ratio"] == 2.0
    assert f["liab_to_assets"] == 0.6
    assert f["gross_margin"] == 0.4
    assert f["rec_to_sales"] == 0.41
    assert f["inv_to_sales"] == 0.1
    assert f["soft_assets_share"] == pytest.approx(0.5)
    assert f["book_to_market"] == 0.5  # 40 / (4 * 20), millions over millions
    assert f["log_assets"] == pytest.approx(math.log(201.0))
    assert f["rect_at"] == pytest.approx(41 / 200)
    assert f["at"] == 200.0


@pytest.mark.parametrize("at", [0.0, -5.0])
def test_nonpositive_assets(at):
    f = F.compute_features(_row(at=at)).iloc[0]
    for c in F.SCALED_FEATURES + ["liab_to_assets", "soft_assets_share"]:
        assert np.isnan(f[c]), c
    assert f["log_assets"] == 0.0


@pytest.mark.parametrize("sale", [0.0, -1.0])
def test_nonpositive_sales(sale):
    f = F.compute_features(_row(sale=sale)).iloc[0]
    for c in ["gross_margin", "rec_to_sales", "inv_to_sales"]:
        assert np.isnan(f[c])
    assert not np.isnan(f["sale_at"])


def test_other_denominators():
    assert np.isnan(F.compute_features(_row(lct=0.0)).iloc[0]["current_ratio"])
    assert np.isnan(F.compute_features(_row(prcc_f=0.0)).iloc[0]["book_to_market"])
    assert np.isnan(F.compute_features(_row(csho=-1.0)).iloc[0]["book_to_market"])


def test_negative_numerators_kept():
    f = F.compute_features(_row(ni=-3.0, at=10.0, ppegt=8.0, che=5.0)).iloc[0]
    assert f["ni_at"] == pytest.approx(-0.3)
    assert f["soft_assets_share"] == pytest.approx(-0.3)  # can go negative


def test_row_independent():
    rows = pd.concat([_row(at=10.0, sale=5.0), _row(at=0.0), _row(at=3.0, sale=-1.0)],
                     ignore_index=True)
    batch = F.compute_features(rows)
    for i in range(len(rows)):
        single = F.compute_features(rows.iloc[[i]])
        pd.testing.assert_frame_equal(batch.iloc[[i]], single)


def test_locked_years_refused():
    r = _row()
    r["fyear"] = 2003
    with pytest.raises(LockedSplitError):
        F.build_features(r)


def test_real_features(raw):
    from footnote.splits import load_split

    ff = load_split("final_fit", raw)
    X = F.build_features(ff)
    assert list(X.columns) == F.FEATURE_NAMES
    assert not np.isinf(X.to_numpy()).any()
    assert X["log_assets"].notna().all()
    assert X[RAW_ITEMS].notna().all().all()
    # Missing exactly where at <= 0 for scaled items.
    bad_at = ff["at"] <= 0
    assert X.loc[bad_at, F.SCALED_FEATURES].isna().all().all()
    assert X.loc[~bad_at, F.SCALED_FEATURES].notna().all().all()
    # Does not use the label or ready-made ratios: changing them changes nothing.
    ff2 = ff.copy()
    ff2["misstate"] = 1 - ff2["misstate"]
    ff2["soft_assets"] = 0.0
    pd.testing.assert_frame_equal(X, F.build_features(ff2))


def test_schema_file(tmp_path):
    import json

    p = F.write_feature_schema(tmp_path / "s.json")
    s = json.loads(p.read_text())
    assert s["n_features"] == 61 and s["features"] == F.FEATURE_NAMES
