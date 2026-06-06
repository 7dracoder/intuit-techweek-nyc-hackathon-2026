# Developer Handoff — SMB Underwriting Challenge

> Read this top-to-bottom to understand the project and continue working on it.
> Written so a new teammate **or an AI assistant (Cursor/Kiro)** can pick it up
> with zero prior context. Companion docs: `PROJECT_GUIDE.md` (what/why of the
> challenge), `ACTION_AND_RESEARCH.md` (status + research topics), `WHAT_WE_DID.md`
> (worker's causal/survival notes).

---

## 0. TL;DR for whoever is continuing

- The project is a 4-deliverable ML credit-underwriting hackathon. Everything
  scored automatically lives in **`solution.py`** → writes 3 CSVs to `submission/`.
- **It works today:** `python solution.py` then `python validate_submission.py ./submission`
  prints **PASS**. Validation AUC ≈ 0.754 (0.7525 final w/ IPW), well-calibrated,
  ~64% approval, 5 engineered features, Deliverable D figures embedded.
- **Do NOT break the two invariants** (Section 7) — they are correctness guarantees
  that took real debugging to get right.
- The remaining upside is in Section 8 (stronger model, richer features, deeper
  causal graph). Pick from there.
- Engine is **scikit-learn only** (no GPU, no LightGBM/libomp). Runs in ~2-5 min
  locally. Colab works but is not required.

---

## 1. What the project is (60-second version)

You are a small-business lender. Using ~100K historical loan applications, you must:

| Deliverable | File | What it is |
|---|---|---|
| **A** | `submission_A_decisions.csv` | Approve/decline each of 13,306 applicants + calibrated probability-of-default (PD) + 90% interval. Scored on **portfolio profit** + calibration. |
| **B** | `submission_B_trajectory.csv` | 13×13 grid: for each origination cohort week × loan-age week, the cumulative default rate of *your approved* loans + 90% interval. Scored on **timing accuracy** + calibration. Must be **monotonic non-decreasing** in age. |
| **C** | `submission_C_counterfactuals.csv` | For 900 queries, the PD **if we `do(feature = value)`** (a causal intervention, not a re-prediction). Scored on **counterfactual accuracy**. |
| **D** | `submission_D_writeup.pdf` | ≤4-page methodology defense. Human-graded; causal section weighted most. |

Hard gate: `validate_submission.py` must print **PASS** or the submission is
disqualified (checks exact names, ID coverage 13,306 / 900 / 169, [0,1] ranges,
interval ordering, B monotonicity).

Full details: `PROJECT_GUIDE.md` and `README.md`.

---

## 2. Repository map

```
.
├── solution.py                     # ★ THE PIPELINE — produces A/B/C (+ assets/metrics.json)
├── make_writeup_assets.py          # offline: DAG/calibration/coverage figures for D (matplotlib)
├── test_solution.py                # unit/sanity tests (incl. no-op invariant) — 6 passing
├── validate_submission.py          # official format gate (do not edit)
├── requirements.txt                # numpy, pandas, scikit-learn (+ matplotlib/pytest for tooling)
├── submission/                     # GENERATED outputs (A/B/C csv)
│   └── assets/                     # GENERATED figures + metrics.json for the writeup
├── submission_D_writeup.md         # filled writeup draft (export to PDF)
├── submission_D_writeup_template.md# original template (keep pristine)
├── dataset/
│   ├── train.csv (85,340)          # history; outcomes only for prior-approved
│   ├── validation.csv (4,489)      # labeled; used to calibrate/tune
│   ├── test.csv (8,817)            # outcomes withheld; scored set
│   ├── data_dictionary.csv         # field, dtype, group, intervenable, notes
│   ├── cohort_week_definitions.csv # cohort_week 1..13 -> date ranges
│   ├── intervention_queries.csv    # 900 do(feature=value) queries for C
│   └── submission_B_template.csv   # the 169-row grid to fill
├── expected_ids/                   # id sets the validator checks against
├── .kiro/specs/causal-and-survival-upgrades/   # spec for the B/C upgrades
├── PROJECT_GUIDE.md  ACTION_AND_RESEARCH.md  WHAT_WE_DID.md
└── DEVELOPER_HANDOFF.md            # this file
```

---

## 3. How to run

```bash
pip install -r requirements.txt        # numpy, pandas, scikit-learn
python solution.py                     # trains everything, writes submission/*.csv
python validate_submission.py ./submission   # must print PASS
python -m pytest test_solution.py -q   # (optional) run the test suite
```

There are **no saved model weights** — the pipeline trains fresh each run
(fast, deterministic via `RANDOM_SEED = 42`). That is intentional: reproducible,
no stale artifacts. If you make it much heavier, consider caching, but today it's
not needed.

---

## 4. Key data facts (memorize these — they drive design)

- **Selection bias:** outcomes (`default_flag`) exist ONLY for loans the prior
  lender approved (train: 51,722 approved / 33,618 declined-no-label). Your model
  learns on an approved-only slice → must generalize to the full population.
- **val + test = exactly the 13,306** scored applicant IDs for Deliverable A.
- All **300 intervention applicants** (900 queries) live in **test.csv**.
- **Recovery is NOT total loss:** loans amortize via daily ACH draws over 60 days;
  median default is day 37, so a defaulter repays ~60% via draws before defaulting.
  `final_recovered_amount` is only the *post-default* ~9%. **Effective recovery ≈ 0.70.**
  (This single insight moved approval from 12.8% → ~62% — the biggest profit lever.)
- `days_to_default` ∈ [1,90], median 37 → real timing signal for B.
- Bank-feed columns are null for ~37% of rows (no linked feed). Don't impute fakes;
  the gradient-boosted trees handle NaN natively and "no feed" is itself signal.
- Categorical columns are integer-coded (per `data_dictionary.csv`); the
  `intervenable` flag marks valid C intervention targets.

---

## 5. Architecture of `solution.py` (function-by-function)

Top of file: paths, `RANDOM_SEED`, toggles (`USE_IPW`), loan economics constants,
and column lists. `main()` runs A → B → C.

### Shared helpers
- `load_data()` — reads all CSVs.
- `get_feature_lists(data_dict)` — returns (feature_cols, categorical_cols),
  excluding outcome/ID/timestamp columns.
- `prepare_features(df, ...)` — coerces features to float (NaN preserved).
- `assign_cohort_week(df, cohorts)` — maps `application_timestamp` → cohort 1..13.
- `train_bagged_models(X, y, cats, sample_weight=None)` — 8 bagged
  `HistGradientBoostingClassifier`s (bootstrap rows + seeds) → point + spread.
- `ensemble_predict(models, X)` — returns (mean_pd, per_member_matrix).
- `fit_isotonic(...)` — isotonic calibration of PD.
- `clamp_intervals(...)` — clamp to [0,1], enforce lo≤point≤hi, floor width.

### Deliverable A — `build_deliverable_A(...)`
1. Train baseline PD model on labeled rows; calibrate on validation.
2. **Reject inference (IPW):** `compute_ipw_weights` fits propensity
   `e(X)=P(approved|X)`, reweights labeled rows by `1/clip(e)`. **Kept only if it
   does not hurt validation AUC/Brier** (auto-toggle).
3. **Recovery model:** `fit_recovery_model` predicts per-loan effective recovery.
4. **Per-bin conformal:** `tune_perbin_conformal` / `apply_perbin_conformal`
   widen intervals per PD bin for ~90% coverage (Mondrian-style).
5. **Decision:** `expected_profit(...)` + a **profit-simulated PD threshold**
   (sweep thresholds against true validation outcomes); approve iff
   `expected_profit > 0 AND PD ≤ threshold`.
6. Writes `submission_A_decisions.csv`; returns artifacts reused by B and C.

### Deliverable B — `build_deliverable_B(...)` (discrete-time hazard survival)
- `expand_person_periods(...)` — explodes each labeled loan into weekly
  person-period rows with a binary event indicator (the discrete-time-survival trick).
- `fit_hazard_models(...)` — bagged hazard classifiers `h(a|x)=P(event at age a)`.
- Per applicant, convert hazards → survival → cumulative default curve; average over
  *your approved* loans per cohort to fill the 13×13 grid; bootstrap for intervals.
- Curves forced **monotone non-decreasing** (validator requirement).
- (An older `estimate_timing_by_risk` risk-bucketed approach is also present.)

### Deliverable C — `build_deliverable_C(...)` (Structural Causal Model)
- `SCM_EDGES` — a domain-justified DAG of **10 structural equations** (child:parents)
  over bank-feed, bureau-credit, and application-context mediators.
- `fit_scm(...)` — fits one `HistGradientBoostingRegressor` per child.
- `scm_intervene(row, feature, value, scm, ...)` — **Pearl's abduction → action →
  prediction**:
  1. **Abduct:** residual `u = actual_child − reg.predict(parents)` per child.
  2. **Act:** set `row[feature] = value`.
  3. **Predict:** re-simulate downstream children in topological order as
     `reg.predict(new_parents) + u`.
  The `+ u` residual is critical — it makes a no-op intervention exactly zero
  (see Section 7). The deterministic ratio is **rescaled**, never recomputed from a
  guessed formula (`_rescale_ratio`).
- Writes `submission_C_counterfactuals.csv`.

---

## 6. Current verified status (last run)

- Validator: **PASS** (0 errors; only the expected "writeup PDF not found" warning).
- **A:** AUC ≈ **0.7544** unweighted / **0.7525** final (IPW kept), Brier ≈ 0.1339,
  near-perfect decile calibration, **100% bin-wise interval coverage at 0.232 mean
  width**, **63.9% approval**. Now **40 features (5 engineered)** — see §8.1.
- **B:** discrete-time hazard survival model, 169-row monotone grid (613,893
  person-period rows, 8 hazard models).
- **C:** 10 fitted structural equations; **no-op interventions move PD by exactly
  0.000** (now asserted in `pytest`: `max|ΔPD| < 1e-9`); directional effects all
  credit-sensible (utilization/overdrafts/delinquency/debt ↑ → PD↑; revenue/cash ↑
  → PD↓); 15.6% up / 84.4% down / 0% near-zero across the 900 queries.
- **Tests:** `python -m pytest test_solution.py -q` → **6 passed**.
- **Deliverable D:** `submission_D_writeup.md` now embeds the SCM DAG, calibration,
  and coverage figures + a results table (figures regenerate via
  `python make_writeup_assets.py`). PDF export + team name remain human steps (§9).

---

## 7. ⚠️ INVARIANTS — do not break these

These are correctness guarantees that took debugging to achieve. Any change to the
SCM or interval logic must preserve them. **Add/keep tests for both.**

1. **No-op invariant (Deliverable C).** `scm_intervene(row, feature, current_value)`
   must change PD by **exactly 0**. If you replace the abduction residual (`+ u`)
   with a raw predicted child value, you reintroduce a regression-to-the-mean
   offset (we measured +0.17 PD on no-ops — a real bug we fixed). Test it:
   set each feature to its own value across a sample of applicants; assert
   `max|Δpd| < 1e-9`.
2. **B monotonicity.** Every cohort's `cumulative_default_rate` (and bounds) must be
   non-decreasing in `loan_age_weeks`, and all values in [0,1] with lo≤point≤hi.
   The validator rejects violations.
3. **The deterministic ratio** `requested_amount_to_observed_revenue` uses an
   unknown internal scale (it's ~annual-based, not monthly). **Never recompute it
   from `requested_amount / observed_monthly_revenue`** — rescale the stored value
   by the multiplicative input change (`_rescale_ratio`). Recomputing corrupts it.

Also: never feed `OUTCOME_COLS`, IDs, or raw `application_timestamp` to any model
(leakage). Keep all three submission files at the exact names/locations.

---

## 8. What to work on next (prioritized roadmap)

Highest leverage first. Each is self-contained.

1. **Stronger PD learner (A, S_P&L).** Try LightGBM/XGBoost (needs
   `brew install libomp` on macOS) and **stack** with the HistGB ensemble.
   Add a time-aware CV hyperparameter search. Keep the IPW toggle + calibration.
2. **Feature engineering (A/B/C). ✅ DONE.** 5 NaN-safe features in
   `compute_engineered`/`attach_engineered` (revenue consistency, debt-service
   coverage, utilization×inquiries, cash-to-requested, prior-default ratio);
   registered into `feature_cols` and recomputed inside `scm_intervene` so the
   no-op invariant holds. Unweighted AUC 0.7517→0.7544. More features welcome.
3. **Deeper / validated causal graph (C, S_C — most-weighted writeup section).**
   Expand `SCM_EDGES` with more defended edges; run causal-discovery sanity checks;
   add sensitivity-to-unobserved-confounding analysis. Keep abduction intact.
4. **Recovery distribution (A).** Model the recovery *distribution* (not just mean)
   so loss uncertainty flows into the decision and intervals.
5. **Calibration polish (S_cal).** Cross-fitted calibration instead of a single
   validation split; tune per-bin conformal widths further.
6. **Writeup (D). ✅ MOSTLY DONE.** `submission_D_writeup.md` matches the current B
   (hazard survival) and C (10-equation SCM) and now embeds the DAG figure,
   calibration plot, coverage table, and a results table (regen via
   `python make_writeup_assets.py`). **Remaining (human):** put the real team name
   and export to `submission/submission_D_writeup.pdf` (≤4 pages, ≥11pt, ≥0.75in);
   verify page count after export and trim if needed.

### Definition of done / "best"
Validator PASS · AUC pushed toward ~0.78 with calibration held · profit-simulated
decision · 10+-equation SCM with abduction · hazard-based B · ~90% bin-wise interval
coverage · complete figure-backed 4-page writeup.

---

## 9. Human-only TODO (cannot be automated)

1. Register the team on the Google Form (see `README.md` / `submission-links.pdf`).
2. Put the real team name in the writeup.
3. Export `submission_D_writeup.md` → `submission/submission_D_writeup.pdf`
   (≤4 pages, ≥11pt font, ≥0.75in margins).
4. Run the validator → confirm **PASS**.
5. Upload all four files (flat folder) to the team's private link.

---

## 10. Git / collaboration notes (read before pushing)

- Remotes: `origin` = team fork (`7dracoder/...`), `upstream` = `intuit/...`
  (challenge source). Work happens on `origin/master`.
- ⚠️ **History was force-pushed once** (the causal/survival upgrade landed as a
  rewritten history with no common ancestor). If you see "no merge base" or
  unexpected divergence after a pull, that's why. **Before any reset/force op,
  create a backup branch** (`git branch backup-$(date +%s)`), then sync.
- Untracked-but-kept files in the working tree: `PROJECT_GUIDE.md`,
  `submission-links.pdf` (and this file until committed). Commit them if you want
  them shared.
- Always re-run `solution.py` + the validator before committing regenerated CSVs,
  and keep `submission_D_writeup_template.md` pristine.
