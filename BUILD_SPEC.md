You are my coding partner for the next 36 hours. We are building Footnote, an accounting-misstatement screen, for the Veritas AI International High School Hackathon (AI + Finance track). I am Dhairya, a solo high school student. The submission deadline is Sunday, November 15, 2026 at 8:59 PM Pacific. It is Saturday morning Pacific now.

Work in an empty git repo I own. I have already set my git identity, so check `git config user.name` and stop only if it is empty. Do not ask me questions unless something blocks you. When you have to choose, take the simpler option and write the choice in DECISIONS.md.

First action: save this entire message, unchanged, to BUILD_SPEC.md in the repo root so you can re-read it after any context reset. Then follow section 12.

## 1 What this is

Footnote takes the 28 financial statement numbers of one US public company for one fiscal year and returns a rank-based score, a percentile against a fixed reference group, a tier, and up to three plain-language areas that pushed the score up. It is a screening aid. It is not a fraud detector.

The research is the main product. The question is whether a boosted model on one year of statements catches more SEC-identified misstatement years than the best one-year rule, including on cases it never saw labeled. An honest "cannot tell apart" is a valid result and the project is built to survive it.

The primary artifact is a notebook, notebooks/footnote.ipynb, that runs the whole experiment from a fresh Colab runtime and writes artifacts/metrics.json. The Streamlit app is second. If time runs short, the notebook plus a one-page write-up is a complete submission, so app work never blocks the notebook.

The exact disclaimer must appear in the app, the README and the notebook: "This project is a prototype and is not financial advice."

## 2 Rules that override everything

1. No test peeking. Until I approve a freeze, rows from fiscal years 2003 to 2014 may be used only for label counts, row counts, missing-value counts and schema checks. No model, rule, threshold, plot or metric may touch their features. Enforce it in code. For example, `load_split("test")` and `load_split("confirmation")` raise an error unless artifacts/frozen.json exists.
2. One test run. After the freeze, score the test and confirmation windows once. If a bug forces a rerun, fix the bug, rerun, and log it in DECISIONS.md. No tuning after seeing any test result.
3. No hand-typed results. Every number in the README, notebook text and app comes from artifacts/metrics.json. Use placeholders such as {{primary_diff}} until the real run exists.
4. Never say a test passed or a notebook ran unless you ran it and the output is visible to me. Anything unverified is labeled "not yet run".
5. Words matter. The positive label is "SEC-identified misstatement year", never "fraud". The score is a ranking, not a probability. Never write "leak-proof", "proves" or "catches fraud". Verdict words come only from the bootstrap interval and are exactly "beats", "cannot tell apart" and "loses to".
6. Never commit the dataset. Download it at run time, verify the SHA-256, and keep data/ in .gitignore.
7. I commit as myself. Do not hide or strip tool attribution. Record AI use honestly in AI_USAGE.md (which tool, what it did, what I reviewed).
8. Cite sources in the README and notebook. Bao et al. (2020), LightGBM and scikit-learn at minimum.

## 3 Experiment card, frozen before any test scoring

````
EXPERIMENT CARD

Question      Does a boosted model on one year of statements catch more SEC-identified
              misstatement years than the best one-year rule, on cases with no labeled
              year in training?
Population    US public company-years in the Bao et al. (2020) file.
Unit          One company-year (one annual filing). Not a company.
Inputs        The 28 raw items from a single fiscal year. Nothing from prior years.
Positive      misstate = 1: a fiscal year covered by an SEC Accounting and Auditing
              Enforcement Release in the source data. Not a court finding. Zeros include
              misstatements never caught.

Windows       Select on: train 1990-1999, validation 2000-2001.
              Final fit: 1990-2001. Gap: 2002 (labels unused). Test: 2003-2008.
              Confirmation: 2009-2014 (never scored before this card; labels incomplete).
Case groups   Seen = case has a positive year in 1990-2001. Unseen = first positive year
              2002 or later. Used only to split test results.

Primary       Recall at a 10% yearly budget on unseen-case positive years:
              model minus best fair rule. Each fiscal year, flag ceil(0.10 * rows).
Secondary     Gap = (model seen - model unseen) - (rule seen - rule unseen), same budget.
Also report   Catches, precision, recall at 1%, 5%, 10% for all positives; PR-AUC and
              ROC-AUC with prevalence; case-level recall; seen vs unseen companies;
              2003-2005 vs 2006-2008; size thirds; the confirmation window.

Baselines     Random (expected value), three one-year rules, same-input logistic regression.
              Best fair rule is chosen on validation and then fixed.
Model         LightGBM, four predeclared settings, chosen on validation recall at the 10%
              yearly budget (tie-break: average precision), after removing validation
              positives from cases that began before 2000.
Thresholds    Tier cutoffs from fiscal-2002 filings scored by the final model. No labels.
Uncertainty   1,000 paired bootstrap resamples of companies. 95% percentile intervals.

Verdict rule  Beats: interval for the difference is entirely above zero.
              Cannot tell: interval includes zero.
              Loses: interval is entirely below zero.
Falsified if  Primary: the interval is not entirely above zero.
              Secondary: the interval for Gap is not entirely above zero.

Disclosure    Before this card existed, an exploratory run scored 2003-2008 with rough
              models and single-ratio rules. It informed the choice of project, not any
              setting below. The seen/unseen split and the confirmation window were not
              looked at.
````

Put the Disclosure paragraph in the notebook and README word for word.

## 4 Data

- File data_FraudDetection_JAR2020.csv from https://raw.githubusercontent.com/JarFraud/FraudDetection/master/data_FraudDetection_JAR2020.csv
- 47,782,071 bytes. SHA-256 ed4cefd9db2a23d0a13911d9f482fc755d3dcf6c9be0bc051af95ddcae5b286f. The loader refuses any file that does not match.
- 146,045 rows, 46 columns, fiscal years 1990 to 2014, 18,444 companies. The pair (gvkey, fyear) never repeats.
- Citation to use as given. Bao, Ke, Li, Yu and Zhang (2020), Journal of Accounting Research 58(1), 199 to 235.
- I could not confirm a license. Treat it as citation required and no redistribution. If the download is blocked from your environment, stop and tell me. I will place the file at data/raw/ myself. As a fallback the notebook may load a copy from my own Google Drive, only if the checksum matches, and the README says so.

Columns
- Keys are fyear and gvkey. Label is misstate. Case id is p_aaer, stored as a decimal number. An empty p_aaer means no case, never one shared group.
- 26 dollar items, in millions of US dollars, no missing values. Several are validly negative (ib, ni, re, ceq, txt). The items are act ap at ceq che cogs dlc dltis dltt dp ib invt ivao ivst lct lt ni ppegt pstk re rect sale sstk txp txt xint.
- csho is millions of shares. It is not dollars.
- prcc_f is dollars per share at fiscal year end. It is not millions.
- That is 28 inputs. The file also has 14 ready-made ratios (dch_wc ch_rsst dch_rec dch_inv soft_assets ch_cs ch_cm ch_roa issue bm dpi reoa EBIT ch_fcf). Most use the prior year. The primary model and the fair rules never use them.

Windows, which you must reproduce exactly

| Window | Fiscal years | Rows | Positive years | Companies | Cases |
| --- | --- | --- | --- | --- | --- |
| Select-train | 1990 to 1999 | 58,634 | 347 | 11,622 | 205 |
| Validation | 2000 to 2001 | 13,114 | 167 | 7,569 | 117 |
| Gap | 2002 | 6,067 | 77 | 6,067 | 77 |
| Test | 2003 to 2008 | 35,166 | 261 | 8,731 | 127 |
| Confirmation | 2009 to 2014 | 33,064 | 112 | 8,411 | 57 |

Other measured facts to reproduce. 70 validation positive rows belong to cases that began before 2000. Of the 261 test positive years, 88 are in seen cases (39 cases) and 173 are unseen. 591 rows have total assets of zero or less. 10,451 rows have sales of zero or less.

## 5 Features, 61 in total, one function, one fiscal year only

- The 28 raw items as given.
- 25 asset-scaled items named `<item>_at`. Every dollar item except `at`, divided by `at`. Missing when `at` is zero or less.
- 8 ratios. current_ratio = act / lct. liab_to_assets = lt / at. gross_margin = (sale - cogs) / sale. rec_to_sales = rect / sale. inv_to_sales = invt / sale. soft_assets_share = (at - ppegt - che) / at. book_to_market = ceq / (prcc_f * csho). log_assets = ln(1 + max(at, 0)). Each is missing when its denominator is zero or less, except log_assets, which is never missing.
- Undefined values are stored as missing, never zero and never infinity. A test asserts no infinite value exists. LightGBM takes missing values directly.
- prcc_f * csho is dollars per share times millions of shares, which is millions of dollars, so it matches ceq.
- soft_assets_share uses gross plant (ppegt) because net plant is not among the 28 inputs. It can go negative. It is not the file's soft_assets column, and the README says so.
- No feature uses another year, another company or the label.

## 6 Baselines

0. Flag nothing. Show once as accuracy with zero catches, to explain why accuracy misleads. Not in budget tables.
1. Random review. Expected catches are the sum over years of K_y times P_y over N_y, with a 95% range from 10,000 hypergeometric draws per year, summed. Label it "expected". Never plot one simulated draw.
2a. Rule, high rect / sale. 2b. Rule, high soft_assets_share (at, ppegt, che). 2c. Rule, average of the two within-year percentile ranks of 2a and 2b. Higher means more suspicious for all three. Missing rule values rank last. Choose the best fair rule by validation recall at the 10% yearly budget, then fix it. Report all three on test. In the notebook, describe the rules as my own domain guess unless I confirm a citation.
3. Logistic regression on the 25 scaled items and 8 ratios. Clip each feature to the 1st and 99th percentile of the fit rows, fill missing with the fit-row median, standardize with the fit-row mean and standard deviation. `LogisticRegression(class_weight="balanced", C=1.0, max_iter=1000)`. No tuning.
4. LightGBM on all 61 features. This is the primary model.
5. Optional and cut first if late. Logistic regression on the 14 ready-made ratios, labeled "extra-input reference". Never call it a fair competitor.

## 7 Model and tuning

- LightGBM 4.7.0. Fixed settings learning_rate=0.03, min_child_samples=50, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, scale_pos_weight equal to negatives over positives in the fit rows, random_state=0.
- Four candidates, nothing else. G1 7 leaves 150 trees. G2 7 leaves 300 trees. G3 15 leaves 150 trees. G4 15 leaves 300 trees.
- Procedure. Fit each candidate once on 1990 to 1999. Score validation 2000 to 2001. Remove the 70 positive rows whose case began before 2000 from the evaluation rows. Pick the highest recall at the 10% yearly budget, tie-break by average precision, then by the smaller model. Write all four rows to artifacts/tuning_log.csv with the metric, runtime and reason for the pick. Refit the winner on 1990 to 2001 with seeds 0 to 4 and average the five predicted scores. That average is the final model.
- Fallback if LightGBM will not install. sklearn HistGradientBoostingClassifier with max_leaf_nodes, max_iter, learning_rate, min_samples_leaf, l2_regularization and class_weight. Record which model ran in metrics.json.
- Earlier timing in my sandbox was about 2.3 seconds for one fit on 58.6k rows, so runtime is not a constraint.

## 8 Evaluation

```
K_y        = ceil(b * N_y)                    filings reviewed in year y
flagged_y  = the K_y rows of year y with the highest score
K          = sum over years of K_y
TP         = flagged rows with misstate = 1
FP         = K - TP
precision  = TP / K
recall     = TP / P                           P = positive years in the window
prevalence = P / N
lift       = precision / prevalence
```

- Budgets are per fiscal year, never pooled across years. Tables use 1%, 5% and 10%. The chart uses a curve from 1% to 20%, computed once and saved.
- Ties. Each row gets a fixed pseudo-random tie key from a seeded shuffle of row ids. Sort by score descending, then tie key. File order never decides a flag.
- A constant score raises an error. Missing scores rank last.
- Group recall. Flags are computed on all rows. Recall for a group of positives such as unseen cases is that group's flagged positives over that group's positives. Do not report precision by group.
- Also report PR-AUC and ROC-AUC next to the prevalence, case-level recall (a case counts as caught if at least one of its positive test years is flagged, so state the definition in the notebook), seen versus unseen companies (a test company is seen if it has any row in 1990 to 2001), 2003 to 2005 versus 2006 to 2008, size thirds (within-year total-assets terciles), and the confirmation window. Log each of these definitions in DECISIONS.md.

Bootstrap, run once on the test window and once on the confirmation window
```
inputs   preds: one row per company-year with gvkey, fyear, y, case_group, tie_key,
         and one score column per method
seed     numpy default_rng(0)
repeat 1,000 times:
    draw len(companies) company ids with replacement
    replicate = every row of each drawn company (a company drawn twice contributes
                two copies, and the copies get distinct tie keys)
    for each method: recompute K_y per year in the replicate, flag, count TP overall
                     and by group
    store each method's recall and every paired difference
        (model - best rule, model - logistic, and the seen/unseen gap)
valid     = replicates whose denominator is above zero
interval  = 2.5th and 97.5th percentile over valid replicates
report    the point estimate from the real data, the interval, and the number of valid replicates
```
Resample companies, not rows, because a company's years are not independent. Score every method on the same replicate (paired). Recompute budgets inside each replicate. Case-level intervals also resample companies, which is slightly off for the 3 two-company cases, and the notebook says so.

Seen and unseen table for the notebook and app: rows are model recall at 10%, best fair rule recall at 10%, and model gap minus rule gap. Columns are seen (88 positive years), unseen (173) and seen minus unseen. Every cell comes from metrics.json, with intervals.

Failure plan. If the primary interval includes zero, the headline is "cannot tell apart". Report that plainly, show the secondary result and what the models catch and miss, and do not change a setting to rescue it.

## 9 Score, tier, drivers and input rules for the scoring code

- Score is the average of the five seed models' raw outputs. It is not a probability and only orders filings.
- Reference cohort is all 6,067 fiscal-2002 filings, scored once by the frozen model using features only. Their labels are never read. Save artifacts/reference_scores.json with 1,001 evenly spaced quantiles of those scores plus the cohort definition. Percentile is the share of reference scores below this score.
- Tiers. Elevated is percentile 95 or higher. Watch is 80 or higher and below 95. No flag is below 80. The shares of the reference cohort are 5%, 15% and 80% by construction.
- Drivers. Average the pred_contrib vectors of the five seed models. Sum contributions within each family below. Keep families whose sum is above zero, sort from largest, show up to three. For each, show one readable ratio and where it sits among 1990 to 2001 filings. If none is above zero, show "No area stood out as pushing this score up."
- Families and the ratio shown. Receivables (rect, rect_at, rec_to_sales; receivables as a share of sales). Inventory (invt, invt_at, inv_to_sales; inventory as a share of sales). Sales and margin (sale, sale_at, cogs, cogs_at, gross_margin; gross margin). Cash and investments (che, ivst, ivao and their _at forms; cash as a share of assets). Plant and soft assets (ppegt, ppegt_at, soft_assets_share; soft-assets share). Liquidity (act, lct, their _at forms, current_ratio; current ratio). Debt and liabilities (dlc, dltt, lt, xint, ap, their _at forms, liab_to_assets; liabilities as a share of assets). Money raised (dltis, sstk and _at forms; new debt and stock as a share of assets). Earnings and taxes (ib, ni, txt, txp, dp and _at forms; net income as a share of assets). Equity and market value (ceq, re, pstk and _at forms, csho, prcc_f, book_to_market; book to market). Size (at, log_assets; total assets). Where a ratio is shown, use the same name for the family in the code and the app.
- Good copy. "Receivables are 41% of sales, higher than 93% of filings from 1990 to 2001. In this model, receivables pushed the score up." Never write that an area "caused X% of fraud risk", never write that a company "is inflating revenue", and never show three lines about one family.

Input guardrails, applied before scoring
- Hard error, no score. Any of the 28 fields is blank or not a number ("All 28 values are needed. Enter 0 for a line the company does not report."). Total assets zero or less ("Total assets must be above zero.").
- Warning, score still runs. Sales zero or less ("Sales-based ratios could not be computed."). Current liabilities, shares or price zero or less (affected ratios become missing). A negative value in act, che, rect, invt, ppegt, lct, lt, ap, cogs or sale ("Check this entry."). Any of the 26 dollar fields above 1,000,000 ("Dollar fields are in millions. This looks like full dollars."), with a button that converts the 26 dollar fields only. Shares and price are never changed, and nothing converts without a click. Liabilities plus equity (lt + ceq) differing from assets by more than 25% of assets ("These don't add up to total assets. Check your entries."). Current assets, cash or receivables larger than total assets ("A part is larger than the whole. Check your entries.").
- Note. Three or more model features outside the 0.5th to 99.5th percentile of 1990 to 2001 filings ("This company is unlike most filings the model learned from, so the score is less reliable."). Always show "Compared with fiscal-2002 filings. Data ends in 2014."
- System. If the model or reference file is missing, no score, and the Evidence tab still loads from metrics.json.

## 10 Stack and layout

Python 3.11, pandas, numpy, scipy, scikit-learn, lightgbm, matplotlib, joblib, pytest, streamlit. Pin versions in requirements.txt after you install them. Use this layout unless you have a strong reason not to.

```
src/footnote/   data.py splits.py features.py baselines.py model.py evaluate.py
                bootstrap.py score.py explain.py guardrails.py
tests/          one test file per module
notebooks/      footnote.ipynb
app/            streamlit_app.py
artifacts/      metrics.json tuning_log.csv reference_scores.json feature_schema.json
                frozen.json (created only at the freeze)
data/           gitignored
README.md  AI_USAGE.md  DECISIONS.md  BUILD_SPEC.md  requirements.txt
```

The notebook calls the functions in src/footnote and shows the tables and charts. It must run top to bottom from a fresh Colab runtime. You cannot check Colab from here, so label it "not yet run in Colab" until I run it.

App styling comes later. Do not design it yet. When we get there, check the installed Streamlit version's theming docs before styling, because I do not want to rely on memory of an older API.

## 11 Phases and gates

Pace targets, not promises. Phases A and B by about noon Pacific Saturday. Freeze and the one test run by about 4 PM. Scoring and explanation code Saturday evening. App and README Sunday morning. Submit by about 6 PM Sunday, which leaves a buffer before 8:59 PM.

Phase A. Repo layout, environment, loader with checksum, windows, case groups, features, and tests for all of it. Gate A. Stop and show me the pytest output and these numbers reproduced from the real file. Window table above, 70, 88 and 173 with 39 seen cases, 591 and 10,451.

Phase B. Baselines, the four LightGBM candidates, validation selection, tuning_log.csv, final refit. Use only 1990 to 2001 features. Gate B. Stop and show me the validation table and which candidate and which rule won. Do not touch 2003 or later. I will approve the freeze. After I approve, write artifacts/frozen.json (selected candidate, best rule, config, git commit hash) and commit it with the message "freeze before test".

Phase C. The one run on the test window and the confirmation window, the bootstrap, metrics.json, the figures, and the notebook assembled around them. Gate C. Stop and show me the headline lines from metrics.json exactly as computed.

Phase D. Reference cohort, scoring function, guardrails, drivers, and their tests. A thin Streamlit shell with a Score tab and an Evidence tab, only if time allows. Phase E. README, AI_USAGE.md, and a final check that no number in any text file was typed by hand.

Between gates, keep going without asking. Commit small and often, with messages that say what changed.

## 12 Your first actions now

1. Check git identity, Python version and network access. Report anything that blocks you in one line.
2. Save this message to BUILD_SPEC.md. Create the layout, .gitignore (with data/), requirements.txt, DECISIONS.md and an AI_USAGE.md stub.
3. Download the CSV into data/raw/, verify the SHA-256, and write the loader.
4. Do Phase A and stop at Gate A.

## 13 Writing style for README, notebook text and app copy

Plain and direct, like a strong high school student or a capable professional. Vary sentence length. No em dashes. Avoid the phrases "deep dive", "leverage", "seamless", "catalyst" and "delve". Do not put colons in headings. Short sentences beat clever ones. Say what the project does and what it does not do.

Make sure to use subagents and work perfectly !!!!!!
