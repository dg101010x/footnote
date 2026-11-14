import math

import pandas as pd
import pytest

from footnote import guardrails as g
from footnote.data import DOLLAR_ITEMS, RAW_ITEMS

# Exact strings as written in BUILD_SPEC.md section 9.
SPEC = {
    "missing": "All 28 values are needed. Enter 0 for a line the company does not report.",
    "assets": "Total assets must be above zero.",
    "sales": "Sales-based ratios could not be computed.",
    "negative": "Check this entry.",
    "dollars": "Dollar fields are in millions. This looks like full dollars.",
    "balance": "These don't add up to total assets. Check your entries.",
    "part": "A part is larger than the whole. Check your entries.",
    "unusual": "This company is unlike most filings the model learned from, so the score is less reliable.",
    "cohort": "Compared with fiscal-2002 filings. Data ends in 2014.",
}


def clean() -> dict[str, object]:
    """A balanced filing that should trigger no error or warning."""
    return {
        "act": 50, "ap": 8, "at": 100, "ceq": 40, "che": 10, "cogs": 80, "dlc": 5,
        "dltis": 3, "dltt": 20, "dp": 4, "ib": 6, "invt": 12, "ivao": 2, "ivst": 1,
        "lct": 25, "lt": 60, "ni": 6, "ppegt": 30, "pstk": 0, "re": 15, "rect": 20,
        "sale": 120, "sstk": 1, "txp": 1, "txt": 2, "xint": 1, "csho": 10, "prcc_f": 12.5,
    }


def codes(result: g.CheckResult) -> set[str]:
    return {m.code for m in result.messages}


def msg(result: g.CheckResult, code: str) -> g.Message:
    return next(m for m in result.messages if m.code == code)


def test_clean_example_only_cohort_note():
    r = g.check_inputs(clean())
    assert r.ok
    assert codes(r) == {"cohort"}
    assert r.values == {k: float(v) for k, v in clean().items()}
    assert not r.suggest_millions_conversion
    assert set(clean()) == set(RAW_ITEMS)


def test_cohort_note_always_present_even_on_error():
    raw = clean()
    raw["at"] = 0
    r = g.check_inputs(raw)
    assert msg(r, "cohort").text == SPEC["cohort"]
    assert msg(r, "cohort").level == "note"


def test_exact_message_texts():
    assert g.MSG_MISSING == SPEC["missing"]
    assert g.MSG_ASSETS == SPEC["assets"]
    assert g.MSG_SALES == SPEC["sales"]
    assert g.MSG_NEGATIVE == SPEC["negative"]
    assert g.MSG_FULL_DOLLARS == SPEC["dollars"]
    assert g.MSG_BALANCE == SPEC["balance"]
    assert g.MSG_PART_WHOLE == SPEC["part"]
    assert g.MSG_UNUSUAL == SPEC["unusual"]
    assert g.MSG_COHORT == SPEC["cohort"]
    assert g.SYSTEM_MISSING_MODEL == (
        "The model or reference file is missing, so no score can be shown. The Evidence tab still works."
    )


def test_no_em_dashes_in_messages():
    texts = [v for k, v in vars(g).items() if k.startswith("MSG_") or k == "SYSTEM_MISSING_MODEL"]
    assert texts and all("—" not in t for t in texts)


@pytest.mark.parametrize("bad", [None, "", "  ", "abc", float("nan"), float("inf"), "-inf", "nan", True])
def test_missing_or_non_numeric_is_error(bad):
    raw = clean()
    raw["rect"] = bad
    values, messages = g.parse_inputs(raw)
    assert values is None
    assert len(messages) == 1
    assert messages[0].level == "error" and messages[0].text == SPEC["missing"]
    assert messages[0].fields == ["rect"]
    r = g.check_inputs(raw)
    assert not r.ok and r.values is None


def test_missing_lists_all_bad_fields():
    raw = clean()
    del raw["csho"]
    raw["sale"] = "twelve"
    _, messages = g.parse_inputs(raw)
    assert set(messages[0].fields) == {"csho", "sale"}


def test_string_parsing_with_commas():
    raw = {k: str(v) for k, v in clean().items()}
    raw["at"] = "1,234.5"
    raw["lt"] = " 740.7 "
    raw["ceq"] = "-12"
    values, messages = g.parse_inputs(raw)
    assert messages == []
    assert values["at"] == 1234.5 and values["lt"] == 740.7 and values["ceq"] == -12.0


@pytest.mark.parametrize("at", [0, -5])
def test_assets_not_positive_is_error(at):
    raw = clean()
    raw["at"] = at
    r = g.check_inputs(raw)
    assert not r.ok and r.values is None
    m = msg(r, "assets_not_positive")
    assert m.level == "error" and m.text == SPEC["assets"] and m.fields == ["at"]
    assert "not_balanced" not in codes(r) and "part_exceeds_whole" not in codes(r)


def test_sales_not_positive_warning():
    raw = clean()
    raw["sale"] = 0
    r = g.check_inputs(raw)
    assert r.ok
    m = msg(r, "sales_not_positive")
    assert m.level == "warning" and m.text == SPEC["sales"]


@pytest.mark.parametrize("field,code,ratio", [
    ("lct", "lct_not_positive", "current ratio"),
    ("csho", "shares_not_positive", "book to market"),
    ("prcc_f", "price_not_positive", "book to market"),
])
def test_denominator_warnings_name_ratio(field, code, ratio):
    raw = clean()
    raw[field] = 0
    r = g.check_inputs(raw)
    assert r.ok
    m = msg(r, code)
    assert m.level == "warning" and ratio in m.text and m.fields == [field]


def test_negative_entries_one_message():
    raw = clean()
    raw["invt"] = -1
    raw["cogs"] = -3
    raw["ni"] = -50  # validly negative, not flagged
    r = g.check_inputs(raw)
    assert r.ok
    negs = [m for m in r.messages if m.code == "negative_value"]
    assert len(negs) == 1
    assert negs[0].text == SPEC["negative"] and negs[0].fields == ["invt", "cogs"]


def test_full_dollars_warning_and_suggestion():
    raw = {k: float(v) * 1_000_000 for k, v in clean().items()}
    raw["csho"] = 10.0
    raw["prcc_f"] = 12.5
    r = g.check_inputs(raw)
    assert r.ok and r.suggest_millions_conversion
    m = msg(r, "full_dollars")
    assert m.text == SPEC["dollars"] and "csho" not in m.fields and "prcc_f" not in m.fields
    assert "at" in m.fields


def test_shares_and_price_above_limit_do_not_trigger_full_dollars():
    raw = clean()
    raw["csho"] = 5_000_000
    raw["prcc_f"] = 2_000_000
    r = g.check_inputs(raw)
    assert "full_dollars" not in codes(r) and not r.suggest_millions_conversion


def test_exactly_one_million_does_not_trigger():
    raw = clean()
    raw["ivao"] = 1_000_000
    assert "full_dollars" not in codes(g.check_inputs(raw))


def test_not_balanced_warning():
    raw = clean()
    raw["lt"] = 90  # 90 + 40 - 100 = 30 > 25
    r = g.check_inputs(raw)
    assert r.ok
    assert msg(r, "not_balanced").text == SPEC["balance"]
    raw["lt"] = 85  # off by exactly 25, not more than 25%
    assert "not_balanced" not in codes(g.check_inputs(raw))


@pytest.mark.parametrize("field", ["act", "che", "rect"])
def test_part_larger_than_whole(field):
    raw = clean()
    raw[field] = 101
    r = g.check_inputs(raw)
    assert r.ok
    m = msg(r, "part_exceeds_whole")
    assert m.text == SPEC["part"] and field in m.fields


def test_unusual_note_threshold():
    assert "unusual_filing" not in codes(g.check_inputs(clean()))
    assert "unusual_filing" not in codes(g.check_inputs(clean(), out_of_range_count=2))
    r = g.check_inputs(clean(), out_of_range_count=3)
    m = msg(r, "unusual_filing")
    assert r.ok and m.level == "note" and m.text == SPEC["unusual"]


def test_ok_false_only_for_errors():
    raw = clean()
    raw.update(sale=-1, lct=0, csho=0, prcc_f=0, lt=500, act=200, ap=2_000_000)
    r = g.check_inputs(raw, out_of_range_count=10)
    assert r.ok
    assert all(m.level != "error" for m in r.messages)
    assert {"sales_not_positive", "lct_not_positive", "negative_value", "full_dollars",
            "not_balanced", "part_exceeds_whole", "unusual_filing", "cohort"} <= codes(r)


def test_convert_to_millions_leaves_shares_and_price():
    values = {k: float(v) * 1_000_000 for k, v in clean().items()}
    values["csho"] = 10.0
    values["prcc_f"] = 12.5
    before = dict(values)
    out = g.convert_to_millions(values)
    assert values == before  # not mutated
    assert out is not values
    assert out["csho"] == 10.0 and out["prcc_f"] == 12.5
    for k in DOLLAR_ITEMS:
        assert math.isclose(out[k], float(clean()[k]))
    assert len(DOLLAR_ITEMS) == 26


def test_count_out_of_range():
    row = pd.Series({"a": 5.0, "b": -1.0, "c": float("nan"), "d": 0.5, "e": 100.0})
    bounds = {"a": (0.0, 1.0), "b": (0.0, 1.0), "c": (0.0, 1.0), "d": (0.0, 1.0),
              "e": (0.0, 100.0), "z": (0.0, 1.0)}
    assert g.count_out_of_range(row, bounds) == 2
