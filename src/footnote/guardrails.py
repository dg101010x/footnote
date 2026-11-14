"""Input guardrails applied before scoring one typed-in filing (spec section 9).

Errors stop the score. Warnings and notes are shown but the score still runs.
Nothing is converted automatically; convert_to_millions runs only on a click.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import pandas as pd

from .data import DOLLAR_ITEMS, RAW_ITEMS

# Exact user-facing strings from the spec.
MSG_MISSING = "All 28 values are needed. Enter 0 for a line the company does not report."
MSG_ASSETS = "Total assets must be above zero."
MSG_SALES = "Sales-based ratios could not be computed."
MSG_LCT = "Current liabilities are zero or less, so the current ratio could not be computed."
MSG_CSHO = "Shares outstanding are zero or less, so book to market could not be computed."
MSG_PRICE = "Price per share is zero or less, so book to market could not be computed."
MSG_NEGATIVE = "Check this entry."
MSG_FULL_DOLLARS = "Dollar fields are in millions. This looks like full dollars."
MSG_BALANCE = "These don't add up to total assets. Check your entries."
MSG_PART_WHOLE = "A part is larger than the whole. Check your entries."
MSG_UNUSUAL = "This company is unlike most filings the model learned from, so the score is less reliable."
MSG_COHORT = "Compared with fiscal-2002 filings. Data ends in 2014."
SYSTEM_MISSING_MODEL = (
    "The model or reference file is missing, so no score can be shown. The Evidence tab still works."
)

NONNEGATIVE_ITEMS = ["act", "che", "rect", "invt", "ppegt", "lct", "lt", "ap", "cogs", "sale"]
PART_ITEMS = ["act", "che", "rect"]
FULL_DOLLAR_LIMIT = 1_000_000.0
BALANCE_TOLERANCE = 0.25
UNUSUAL_MIN_COUNT = 3

Level = Literal["error", "warning", "note"]


@dataclass
class Message:
    """One message for the user, with the input fields it concerns."""

    level: Level
    code: str
    text: str
    fields: list[str] = field(default_factory=list)


@dataclass
class CheckResult:
    """Outcome of check_inputs. ok is False when any message is an error."""

    ok: bool
    messages: list[Message]
    values: dict[str, float] | None
    suggest_millions_conversion: bool = False


def _to_float(value: object) -> float | None:
    """Parse one entry to a finite float, or None if blank, non-numeric, NaN or infinite."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, str):
        text = value.strip().replace(",", "")
        if not text:
            return None
        try:
            out = float(text)
        except ValueError:
            return None
    else:
        try:
            out = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None
    return out if math.isfinite(out) else None


def parse_inputs(raw: dict[str, object]) -> tuple[dict[str, float] | None, list[Message]]:
    """Parse all 28 inputs. Returns (values, []) or (None, [error naming the bad fields])."""
    values: dict[str, float] = {}
    bad: list[str] = []
    for name in RAW_ITEMS:
        parsed = _to_float(raw.get(name))
        if parsed is None:
            bad.append(name)
        else:
            values[name] = parsed
    if bad:
        return None, [Message("error", "missing_or_not_number", MSG_MISSING, bad)]
    return values, []


def check_inputs(raw: dict[str, object], out_of_range_count: int | None = None) -> CheckResult:
    """Apply every input rule from spec section 9 and collect the messages."""
    values, messages = parse_inputs(raw)
    suggest = False
    if values is not None:
        v = values
        at_ok = v["at"] > 0
        if not at_ok:
            messages.append(Message("error", "assets_not_positive", MSG_ASSETS, ["at"]))
        if v["sale"] <= 0:
            messages.append(Message("warning", "sales_not_positive", MSG_SALES, ["sale"]))
        if v["lct"] <= 0:
            messages.append(Message("warning", "lct_not_positive", MSG_LCT, ["lct"]))
        if v["csho"] <= 0:
            messages.append(Message("warning", "shares_not_positive", MSG_CSHO, ["csho"]))
        if v["prcc_f"] <= 0:
            messages.append(Message("warning", "price_not_positive", MSG_PRICE, ["prcc_f"]))
        negative = [c for c in NONNEGATIVE_ITEMS if v[c] < 0]
        if negative:
            messages.append(Message("warning", "negative_value", MSG_NEGATIVE, negative))
        large = [c for c in DOLLAR_ITEMS if abs(v[c]) > FULL_DOLLAR_LIMIT]
        if large:
            suggest = True
            messages.append(Message("warning", "full_dollars", MSG_FULL_DOLLARS, large))
        if at_ok:
            if abs(v["lt"] + v["ceq"] - v["at"]) > BALANCE_TOLERANCE * v["at"]:
                messages.append(Message("warning", "not_balanced", MSG_BALANCE, ["lt", "ceq", "at"]))
            parts = [c for c in PART_ITEMS if v[c] > v["at"]]
            if parts:
                messages.append(Message("warning", "part_exceeds_whole", MSG_PART_WHOLE, parts + ["at"]))
    if out_of_range_count is not None and out_of_range_count >= UNUSUAL_MIN_COUNT:
        messages.append(Message("note", "unusual_filing", MSG_UNUSUAL, []))
    messages.append(Message("note", "cohort", MSG_COHORT, []))
    ok = not any(m.level == "error" for m in messages)
    return CheckResult(ok=ok, messages=messages, values=values if ok else None,
                       suggest_millions_conversion=suggest)


def convert_to_millions(values: dict[str, float]) -> dict[str, float]:
    """Return a new dict with the 26 dollar fields divided by 1,000,000. Shares and price unchanged."""
    out = dict(values)
    for name in DOLLAR_ITEMS:
        if name in out:
            out[name] = float(out[name]) / 1_000_000.0
    return out


def count_out_of_range(features_row: pd.Series, bounds: dict[str, tuple[float, float]]) -> int:
    """Count model features that are present and outside their [lo, hi] bounds."""
    count = 0
    for name, (lo, hi) in bounds.items():
        if name not in features_row.index:
            continue
        value = features_row[name]
        if pd.isna(value):
            continue
        if value < lo or value > hi:
            count += 1
    return count
