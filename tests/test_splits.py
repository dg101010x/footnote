import json

import pandas as pd
import pytest

from footnote import splits
from footnote.splits import LockedSplitError


EXPECTED_TABLE = {
    "select_train": (58_634, 347, 11_622, 205),
    "validation": (13_114, 167, 7_569, 117),
    "gap": (6_067, 77, 6_067, 77),
    "test": (35_166, 261, 8_731, 127),
    "confirmation": (33_064, 112, 8_411, 57),
}


def _toy():
    return pd.DataFrame(
        {
            "fyear": [1995, 2001, 2002, 2003, 2010],
            "gvkey": [1, 1, 2, 2, 3],
            "misstate": [1, 1, 0, 1, 0],
            "p_aaer": [10.0, 10.0, None, 20.0, None],
            "row_id": range(5),
            "tie_key": [4, 3, 2, 1, 0],
            "at": [1.0] * 5,
        }
    )


@pytest.mark.parametrize("name", ["test", "confirmation"])
def test_locked_windows_raise(name):
    with pytest.raises(LockedSplitError):
        splits.load_split(name, _toy())


def test_unlocked_windows_ok():
    df = _toy()
    assert len(splits.load_split("select_train", df)) == 1
    assert len(splits.load_split("final_fit", df)) == 2
    assert len(splits.load_split("gap", df)) == 1


def test_assert_unlocked():
    with pytest.raises(LockedSplitError):
        splits.assert_unlocked(_toy())
    splits.assert_unlocked(_toy().query("fyear <= 2002"))


FAKE_FREEZE = {"selected_candidate": "G1", "best_rule": "2a", "config": {}, "commit": "x"}


@pytest.mark.parametrize("content", ["", "not json", json.dumps({"frozen": True})])
def test_invalid_freeze_file_does_not_unlock(_not_frozen, content):
    _not_frozen.write_text(content)
    with pytest.raises(LockedSplitError):
        splits.load_split("test", _toy())


def test_freeze_unlocks(_not_frozen):
    _not_frozen.write_text(json.dumps(FAKE_FREEZE))
    assert len(splits.load_split("test", _toy())) == 1
    splits.assert_unlocked(_toy())


def test_case_groups_toy():
    grp = splits.case_group(_toy())
    assert list(grp) == ["seen", "seen", "", "unseen", ""]


def test_window_table_real(raw):
    table = splits.window_summary(splits.label_view(raw))
    for name, (rows, pos, comp, cases) in EXPECTED_TABLE.items():
        r = table.loc[name]
        assert (r.rows, r.positive_years, r.companies, r.cases) == (rows, pos, comp, cases), name


def test_gate_a_facts_real(raw):
    f = splits.gate_a_facts(raw)
    assert f["validation_pos_from_cases_before_2000"] == 70
    assert f["test_positive_years"] == 261
    assert f["test_seen_positive_years"] == 88
    assert f["test_unseen_positive_years"] == 173
    assert f["test_seen_cases"] == 39
    assert f["rows_at_le_0"] == 591
    assert f["rows_sale_le_0"] == 10_451
    assert f["two_company_cases"] == 3


def test_label_view_has_no_features(raw):
    lv = splits.label_view(raw)
    assert set(lv.columns) == set(splits.LABEL_COLUMNS)


def test_real_locked(raw):
    with pytest.raises(LockedSplitError):
        splits.load_split("test", raw)
    with pytest.raises(LockedSplitError):
        splits.load_split("confirmation", raw)
    ff = splits.load_split("final_fit", raw)
    assert ff["fyear"].max() == 2001 and len(ff) == 58_634 + 13_114


def test_load_raw_masks_locked_values(raw):
    from footnote.data import RAW_ITEMS, READY_RATIOS

    locked = raw["fyear"] >= 2003
    assert raw.attrs["locked_values_masked"]
    assert raw.loc[locked, RAW_ITEMS + READY_RATIOS].isna().all().all()
    assert raw.loc[~locked, RAW_ITEMS].notna().all().all()
    # Labels survive for counts.
    assert raw.loc[locked, "misstate"].sum() == 261 + 112
