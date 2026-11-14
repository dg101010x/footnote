import numpy as np
import pandas as pd
import pytest

from footnote import evaluate as ev


def _preds(fyear, y, score, tie_key, case_id=None, case_group=None):
    n = len(fyear)
    return pd.DataFrame(
        {
            "gvkey": np.arange(n),
            "fyear": fyear,
            "y": y,
            "case_id": case_id if case_id is not None else [np.nan] * n,
            "case_group": case_group if case_group is not None else [""] * n,
            "tie_key": tie_key,
            "s": score,
        }
    )


def test_budget_k_float_safe():
    assert ev.budget_k(130, 0.10) == 13
    assert ev.budget_k(131, 0.10) == 14
    assert ev.budget_k(100, 0.01) == 1
    assert ev.budget_k(101, 0.01) == 2
    assert ev.budget_k(7, 0.0) == 0
    assert ev.budget_k(20, 0.05) == 1


def test_rank_order_ties_by_tie_key_and_nan_last():
    score = [0.5, np.nan, 0.9, 0.5, 0.1]
    tie = [9, 0, 5, 2, 1]
    # 0.9 first, then the two 0.5 rows by tie key (row 3 key 2 before row 0 key 9), 0.1, NaN.
    assert ev.rank_order(score, tie).tolist() == [2, 3, 0, 4, 1]


@pytest.mark.parametrize("score", [[1.0, 1.0, 1.0], [np.nan, np.nan], [2.0, np.nan, 2.0]])
def test_constant_score_raises(score):
    with pytest.raises(ValueError, match="constant score"):
        ev.rank_order(score, np.arange(len(score)))
    with pytest.raises(ValueError, match="constant score"):
        ev.flag_by_budget([2000] * len(score), score, np.arange(len(score)), 0.5)


def test_flag_ties_use_tie_key_not_file_order():
    # Four tied rows, budget 0.5 -> K = 2. Lowest tie keys are rows 3 and 1.
    flags = ev.flag_by_budget([2000] * 5, [1.0, 1.0, 0.0, 1.0, 1.0], [7, 2, 0, 1, 9], 0.4)
    assert flags.tolist() == [False, True, False, True, False]


def test_flag_nan_ranks_last():
    flags = ev.flag_by_budget([2000] * 4, [np.nan, 0.2, 0.1, 0.3], [0, 1, 2, 3], 0.75)
    assert flags.tolist() == [False, True, True, True]


def test_flag_per_year_ceil_including_130_rows():
    rng = np.random.default_rng(3)
    fy = np.r_[[2003] * 130, [2004] * 11]
    score = rng.random(fy.size)
    flags = ev.flag_by_budget(fy, score, rng.permutation(fy.size), 0.10)
    assert flags[fy == 2003].sum() == 13
    assert flags[fy == 2004].sum() == 2  # ceil(1.1)
    # Within each year the flagged rows are the top scores of that year.
    for yr in (2003, 2004):
        m = fy == yr
        assert score[m][flags[m]].min() > score[m][~flags[m]].max()


def test_years_not_pooled():
    # Year 2000 has all the high scores; pooled flagging would pick only 2000 rows.
    fy = [2000] * 4 + [2001] * 4
    score = [10, 11, 12, 13, 1, 2, 3, 4]
    flags = ev.flag_by_budget(fy, score, np.arange(8), 0.25)
    assert flags.tolist() == [False, False, False, True, False, False, False, True]


def test_flag_matches_lexsort_reference_on_random_data():
    rng = np.random.default_rng(0)
    n = 2000
    fy = rng.integers(2003, 2009, n)
    score = np.round(rng.random(n), 2)  # many ties
    score[rng.random(n) < 0.05] = np.nan
    tie = rng.permutation(n)
    flags = ev.flag_by_budget(fy, score, tie, 0.1)
    key = -score
    key[np.isnan(key)] = np.inf
    order = np.lexsort((tie, key, fy))
    ref = np.zeros(n, dtype=bool)
    for yr in np.unique(fy):
        rows = order[fy[order] == yr]
        ref[rows[: ev.budget_k(rows.size, 0.1)]] = True
    assert (flags == ref).all()


def _toy():
    # Two years of 10 rows; budget 0.2 flags 2 per year.
    fy = [2000] * 10 + [2001] * 10
    score = list(range(10, 0, -1)) + list(range(10, 0, -1))
    y = [1, 0, 1, 0, 0, 0, 0, 0, 0, 0] + [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
    case = [1.0, np.nan, 2.0] + [np.nan] * 12 + [1.0] + [np.nan] * 4
    grp = ["seen", "", "unseen"] + [""] * 12 + ["seen"] + [""] * 4
    return _preds(fy, y, score, np.arange(20), case, grp)


def test_budget_metrics_hand_computed():
    m = ev.budget_metrics(_toy(), "s", 0.2)
    # Flags: rows 0,1 (2000) and 10,11 (2001). TP = row 0 only.
    assert (m["K"], m["TP"], m["FP"], m["P"], m["N"]) == (4, 1, 3, 3, 20)
    assert m["precision"] == pytest.approx(0.25)
    assert m["recall"] == pytest.approx(1 / 3)
    assert m["prevalence"] == pytest.approx(0.15)
    assert m["lift"] == pytest.approx(0.25 / 0.15)


def test_recall_nan_without_positives():
    p = _toy()
    p["y"] = 0
    assert np.isnan(ev.budget_metrics(p, "s", 0.2)["recall"])


def test_group_and_case_recall():
    p = _toy()
    flags = ev.flag_by_budget(p["fyear"], p["s"], p["tie_key"], 0.2)
    seen = ev.group_recall(p, flags, p["case_group"] == "seen")
    assert seen == {"positives": 2, "caught": 1, "recall": 0.5}
    unseen = ev.group_recall(p, flags, p["case_group"] == "unseen")
    assert unseen == {"positives": 1, "caught": 0, "recall": 0.0}
    # Case 1 has a flagged row (row 0), case 2 does not.
    assert ev.case_recall(p, flags) == {"cases": 2, "caught": 1, "recall": 0.5}


def test_auc_with_nan_scores_ranked_lowest():
    y = [1, 0, 1, 0]
    m = ev.auc_metrics(y, [0.9, 0.5, np.nan, 0.1])
    # NaN becomes lowest: order 0.9(1), 0.5(0), 0.1(0), NaN(1).
    assert m["roc_auc"] == pytest.approx(0.5)
    assert m["pr_auc"] == pytest.approx((1.0 + 0.5) / 2)
    assert m["prevalence"] == 0.5


def test_recall_curve_columns_and_budgets():
    c = ev.recall_curve(_toy(), "s")
    assert list(c.columns) == ["budget", "K", "TP", "recall", "precision"]
    assert len(c) == 20 and c["budget"].iloc[0] == 0.01 and c["budget"].iloc[-1] == 0.2
    assert c["recall"].is_monotonic_increasing


def test_random_expected_formula():
    fy = [2000] * 130 + [2001] * 50
    y = [1] * 13 + [0] * 117 + [1] * 5 + [0] * 45
    p = _preds(fy, y, np.arange(180), np.arange(180))
    r = ev.random_expected(p, 0.10, n_draws=2000, seed=0)
    expected = 13 * 13 / 130 + 5 * 5 / 50
    assert r["expected_TP"] == pytest.approx(expected)
    assert r["expected_recall"] == pytest.approx(expected / 18)
    assert r["lo"] <= expected <= r["hi"]
    assert r["label"] == "expected"
    assert ev.random_expected(p, 0.10, n_draws=2000, seed=0) == r


def test_flag_nothing():
    assert ev.flag_nothing(_toy()) == {"accuracy": 0.85, "TP": 0, "N": 20, "P": 3}


def test_evaluate_method_keys():
    out = ev.evaluate_method(_toy(), "s")
    assert {"b01", "b05", "b10", "pr_auc", "roc_auc", "prevalence"} <= set(out)
    assert out["b10"]["K"] == 2


def test_period_half():
    assert list(ev.period_half([2003, 2005, 2006, 2008, 2009])) == [
        "2003-2005", "2003-2005", "2006-2008", "2006-2008", ""
    ]


def test_size_thirds_within_year():
    p = _preds([1995] * 6 + [1996] * 3, [0] * 9, np.arange(9), np.arange(9))
    p["at"] = [6, 5, 4, 3, 2, 1, 100, 300, 200]
    assert ev.size_thirds(p).tolist() == [
        "large", "large", "mid", "mid", "small", "small", "small", "large", "mid"
    ]


def test_subgroup_masks():
    p = _toy()
    p["fyear"] = [2003] * 10 + [2007] * 10
    masks = ev.subgroup_masks(p, {"company_seen": np.r_[np.ones(5), np.zeros(15)]})
    assert masks["case_seen"].sum() == 2 and masks["case_unseen"].sum() == 1
    assert masks["period_2003-2005"].sum() == 10 and masks["period_2006-2008"].sum() == 10
    assert masks["company_seen"].sum() == 5
    assert "size_small" not in masks
