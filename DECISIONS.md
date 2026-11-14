# Decisions

Choices made during the build, simplest option first. Newest at the bottom of each section.

## Environment

- Git identity in this cloud session reads `user.name = Claude`, `user.email = noreply@anthropic.com`. It is not empty, so per the spec the build continues. Dhairya should check this before the final commits, since the spec says commits are his.
- Build environment uses Python 3.11.x in a local `.venv`. Versions in requirements.txt are the ones installed and tested here. Colab may ship other versions. The notebook will install from requirements.txt.
- Tests find the package through `pythonpath = src` in pytest.ini. No packaging step.

## Data

- The CSV downloaded from the GitHub URL in the spec matched the expected size (47,782,071 bytes) and SHA-256. The loader verifies both and refuses any other file. Download goes to a `.part` file first and is only renamed after it verifies.
- `row_id` is the 0-based position in the file. `tie_key` is `numpy.random.default_rng(0).permutation(n_rows)` over row ids. It is the only tie breaker after the score.
- Schema checks run on all years: shape, columns, unique (gvkey, fyear), no missing values in the 28 inputs, p_aaer present exactly on positive rows.

## Windows and the lock

- Lock enforcement has two layers. `splits.load_split("test" | "confirmation")` raises `LockedSplitError` unless `artifacts/frozen.json` exists. `splits.assert_unlocked()` raises if any row is from 2003 or later before the freeze, and `features.build_features()` calls it. Later modules (baselines, model, evaluate) will call it too.
- `splits.label_view()` returns only keys, label, case id, row id and tie key. It is the only view of 2003 to 2014 rows before the freeze.
- Tests point `FOOTNOTE_FROZEN_PATH` at a temporary path so they always run as pre-freeze unless a test creates the file.
- The counts of rows with total assets <= 0 (591) and sales <= 0 (10,451) are over all 146,045 rows, as the spec states them. These are schema-style counts of two raw inputs and feed no model, rule or threshold.
- The "no infinite value" test runs on 1990 to 2001 features only, because building features for 2003 or later is locked until the freeze.

## Case groups

- A case is one p_aaer value among positive rows. A case "begins" in its first positive fiscal year.
- Seen case: first positive year is 2001 or earlier (so it has a positive year in 1990 to 2001). Unseen case: first positive year is 2002 or later. Case groups are computed from labels of all years, which is a label count.
- Validation model selection drops positive validation rows whose case began before 2000 (70 rows).
- A test company is "seen" if it has any row in 1990 to 2001 (from the label view only).
- 3 cases span two companies (gvkey). Case-level bootstrap intervals that resample companies are slightly off for these.
