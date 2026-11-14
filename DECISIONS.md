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

## Lock hardening after the Phase A audit

An independent subagent audit found three holes. All three are fixed.

- `load_raw()` now blanks out (NaN) the 28 inputs and the 14 ready-made ratios for fiscal years 2003 to 2014 whenever there is no valid freeze. The schema checks and the all-row counts (assets <= 0, sales <= 0, missing values per column) run before the blanking and are kept in `df.attrs["schema_counts"]`. After the freeze, reload to get the real values.
- `compute_features()` checks the lock whenever the frame has an `fyear` column. A single row typed into the app has no `fyear`, so it is not blocked. `build_features()` requires `fyear`.
- `is_frozen()` no longer trusts any file that exists. The file must be valid JSON with `selected_candidate`, `best_rule`, `config` and `commit`.

## Phase B and later, process

- On Saturday Dhairya wrote "GO ahead start phase B and all the other phases", then "just make sure you use sub agents to build". The lead takes this as approval to write `artifacts/frozen.json` once validation selection is finished, without stopping at Gate B. The validation table and the picks are still reported to him unchanged, and are saved in `artifacts/tuning_log.csv` and `artifacts/validation_table.csv` before any test scoring.
- Modules are built by parallel subagents, each working on its own files. The lead (main session) reviews, runs the tests, commits, writes the freeze and runs the one test-window scoring.

## Baselines and selection

- Rule 2c: the within-year percentile rank (pct rank among non-missing values in that fiscal year) of 2a and of 2b, averaged. A missing component counts as percentile 0, so rows missing both rank at the bottom. Rules 2a and 2b keep NaN, and NaN ranks last in evaluation.
- The best fair rule is chosen on the same validation evaluation rows as the model, meaning 2000 to 2001 with the 70 positive rows from cases that began before 2000 removed. Yearly budgets are computed on those remaining rows.
- The logistic baseline scores with `decision_function` (it is a ranking). Order of preprocessing: clip to fit-row 1st/99th percentiles, then fill missing with the fit-row median, then standardize with the fit-row mean and standard deviation (a standard deviation of 0 becomes 1).
- For model selection the logistic baseline is fit on 1990 to 1999 and shown on validation for information only. For the test run it is refit on 1990 to 2001, the same rows as the final model.
- Final LightGBM boosters are saved as LightGBM text files in `artifacts/model/`, not pickles, so they load across versions and are safe to commit. The model is not the dataset.
- Budget K_y = ceil(b * N_y), computed as ceil(round(b * N_y, 9)) so floating-point error cannot add a row.

## Guardrails (built by a subagent, reviewed by the lead)

- The spec gives no wording for the current-liabilities, shares and price warnings. They read "Current liabilities are zero or less, so the current ratio could not be computed.", "Shares outstanding are zero or less, so book to market could not be computed." and "Price per share is zero or less, so book to market could not be computed."
- The full-dollars check fires when the absolute value is above 1,000,000. Exactly 1,000,000 does not fire.
- When total assets are zero or less, the balance check and the part-larger-than-whole check are skipped. The error already blocks the score.
- When any field is missing or not a number, only that error and the cohort note are returned.
- The balance check fires only when the gap is more than 25% of assets, not when it equals 25%.
- `count_out_of_range` skips missing features and features that have no bounds.

## Baselines and model (built by a subagent, reviewed by the lead)

- `FinalModel` keeps each seed's LightGBM booster and always scores with `Booster.predict`, so a fitted model and a reloaded model run the same code. Tests check that save then load gives the same scores and contributions, and that contributions plus bias equal the score.
- `model_meta.json` lists the seeds separately and records scale_pos_weight from the fit rows.
- If both parts of rule 2c are missing, the rule 2c score is 0, not missing, so the row sits at the bottom of its year.
- The logistic standard deviation uses ddof=0. `scale_pos_weight` raises an error when the fit rows have no positives.
- In candidate selection, "smaller model" means fewer leaves times trees, then fewer leaves.
