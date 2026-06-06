# Action & Research Plan — Road to the Best Submission

> This file is the honest, complete status of the solution: what is **fully done**,
> what is **only partially done** (and why), what is **not started**, exactly
> **where in the code** each change goes, **what you need to do by hand**, and
> **what to research yourself** so you can defend every choice.

Read this together with `PROJECT_GUIDE.md` (the what/why of the project) and
`solution.py` (the implementation).

---

## Legend

- ✅ **DONE** — implemented, tested, working in `solution.py`.
- 🟡 **PARTIAL** — a basic version works, but it is the simple/safe version and a
  stronger version is needed for top marks.
- 🔴 **NOT STARTED** — planned, not yet built.
- 👤 **YOUR JOB** — a human step Kiro cannot do for you.

---

## 1. Status of every piece

| # | Component | Deliverable | Status | Where in `solution.py` |
|---|---|---|---|---|
| 1 | Data loading + feature/categorical detection | all | ✅ DONE | `load_data`, `get_feature_lists`, `prepare_features` |
| 2 | Cohort-week assignment from timestamp | B | ✅ DONE | `assign_cohort_week` |
| 3 | PD model (bagged gradient boosting) | A | ✅ DONE | `train_bagged_models`, `ensemble_predict` |
| 4 | Probability calibration (isotonic) | A | ✅ DONE | `fit_isotonic` |
| 5 | Effective-recovery economics fix | A | ✅ DONE | `build_deliverable_A` (recovery block) |
| 6 | Profit decision + simulated threshold | A | ✅ DONE | `expected_profit`, threshold sweep in `build_deliverable_A` |
| 7 | Selection-bias correction (IPW, measured toggle) | A | ✅ DONE | `compute_ipw_weights`, `build_deliverable_A` |
| 8 | Per-loan recovery (LGD) model | A | ✅ DONE | `fit_recovery_model` |
| 9 | Risk-bucketed default-timing curves | B | ✅ DONE | `estimate_timing_by_risk` |
| 10 | Cohort incidence + bootstrap intervals | B | ✅ DONE | `build_deliverable_B` |
| 11 | SCM counterfactuals (abduction-action-prediction) | C | ✅ DONE | `fit_scm`, `scm_intervene`, `build_deliverable_C` |
| 12 | Per-bin (Mondrian) conformal intervals | A/B/C | ✅ DONE | `tune_perbin_conformal`, `apply_perbin_conformal` |
| 13 | Stronger learner (LightGBM/stacking) | A/B/C | 🔴 NOT STARTED | `CLF_PARAMS`, `train_bagged_models` |
| 14 | Feature engineering (5 NaN-safe features) | A/B/C | ✅ DONE | `compute_engineered`, `attach_engineered`, `main` |
| 14b | Run diagnostics + writeup figures | D | ✅ DONE | `METRICS` dump, `make_writeup_assets.py` |
| 14c | Invariant tests (no-op, monotonicity, FE) | all | ✅ DONE | `test_solution.py` (6 passing) |
| 15 | Writeup (Deliverable D) — figures + tables embedded | D | 🟡 PARTIAL (team name + PDF export = 👤) | `submission_D_writeup.md` |
| 16 | Team registration | — | 👤 YOUR JOB | Google Form |
| 17 | Export writeup → PDF | D | 👤 YOUR JOB | `submission/submission_D_writeup.pdf` |
| 18 | Upload 4 files | — | 👤 YOUR JOB | team private link |

**Verified after the upgrades + feature engineering:** validator prints **PASS**;
validation AUC **0.7544** unweighted / **0.7525** final (IPW kept), Brier 0.1339,
near-perfect calibration, **100% bin-wise interval coverage at 0.232 width**,
**63.9% approval**; 40 features (5 engineered); SCM no-op interventions move PD by
exactly 0.000 (`pytest` asserts `max|dPD| < 1e-9`) and real interventions are
directionally correct (15.6% up / 84.4% down / 0% near-zero — the query set is
dominated by improvement interventions). 6/6 tests pass.

---

## 2. The PARTIAL items explained (what's missing and why it matters)

### 🟡 #6 — Profit-based decision rule
**What's done:** Approve iff `(1-PD)·income − PD·loss > 0`, using the analytic
break-even (~24% PD).
**What's missing:** The threshold is computed from a formula, not *verified against
realized profit*. The better version runs a **portfolio simulation** on validation:
try many thresholds, compute total realized profit for each, pick the best.
**Why it matters:** Squeezes extra profit (S_P&L) near the decision boundary where
approvals flip.
**Where:** add a `simulate_profit(threshold)` loop in `build_deliverable_A`, choose
the threshold that maximizes validation profit, then apply it.

### 🟡 #9 — Default-timing curve F(a)
**What's done:** One **global** curve — fraction of all defaults occurring by week
`a`, applied to every cohort.
**What's missing:** A **feature-conditioned hazard / survival model** so each cohort
gets its own shape (a riskier cohort defaults *faster*, not just *more*).
**Why it matters:** Directly improves trajectory accuracy (S_traj) and gives tighter
B intervals (S_cal).
**Where:** replace `estimate_timing_fraction` with a discrete-time hazard model
(see Research §4.3) and feed per-cohort curves into `build_deliverable_B`.

### 🟡 #10 — Cohort incidence intervals
**What's done:** Intervals from **bootstrapping the mean PD** of approved loans in
each cohort.
**What's missing:** This only captures sampling noise in the *incidence*, not model
uncertainty in the *timing* curve. A survival model would propagate both.
**Where:** `build_deliverable_B` interval block.

### 🟡 #11 — Counterfactual (Deliverable C) — THE big one
**What's done:** Set the intervened feature, recompute the **one deterministic
mediator** (`requested_amount_to_observed_revenue`), re-score. Better than naive
re-prediction.
**What's missing:** A real **Structural Causal Model (SCM)**. Right now, intervening
on (say) revenue does NOT ripple to cash balance, overdrafts, utilization, etc. —
only to the one ratio. A full SCM re-simulates *all* downstream children.
**Why it matters MOST:** S_C explicitly penalizes naive re-prediction, and the
writeup's causal section is the **highest-weighted** of all. This is where the
contest is won.
**Where:** new `fit_scm()` + `intervene()` functions; `build_deliverable_C` calls
them instead of `recompute_engineered`.

### 🟡 #12 — Uncertainty intervals
**What's done:** Ensemble spread + a **single additive** conformal widening to hit
~90% coverage on average.
**What's missing:** **Per-risk-bin (Mondrian) conformal** widening, so coverage is
right *across the whole PD range*, not just on average → narrower intervals at the
same coverage.
**Where:** generalize `tune_conformal_width` to return a per-bin delta; apply it
by bin in A and C.

### 🟡 #15 — Writeup
**What's done:** Full 5-section draft in `submission_D_writeup.md`.
**What's missing:** Team name, a DAG figure, a calibration plot, a coverage table,
and a results table — plus updates after any P1–P4 work lands. Then export to PDF.

---

## 3. The NOT-STARTED items (the roadmap, in priority order)

### 🔴 P1 — Structural Causal Model for Deliverable C  *(highest leverage)*
The flagship upgrade. Turns C from "edit one cell" into "intervene and propagate."
See Research §4.1. Touches: new SCM functions, `build_deliverable_C`, the writeup.

### 🔴 P2 — Selection-bias correction (IPW) for Deliverable A
Reweight training rows by inverse propensity-of-approval so the model generalizes
to the never-funded population. See Research §4.2. **Run it as a toggle and keep it
only if validation profit/calibration improves.** Touches: new `compute_ipw_weights`,
`train_bagged_models` (add `sample_weight`), `build_deliverable_A`.

### 🔴 P2b — Per-loan recovery (LGD) model
Replace the single ~70% portfolio recovery with a model predicting recovery per
loan from features. Sharpens the profit decision near break-even. Touches: new
`fit_recovery_model`, `expected_profit_decision`.

### 🔴 P3 — Feature-conditioned survival model for B
See §9 above and Research §4.3.

### 🔴 P4 — Per-bin conformal calibration
See §12 above and Research §4.4.

### 🔴 P5 — Stronger learner + feature engineering
Install LightGBM (`brew install libomp` then `pip install lightgbm`), stack it with
the HistGB ensemble; add engineered features (revenue consistency = stated vs
observed, debt-service coverage, prior-default ratio, bank-feed-linked interactions).
Touches: `LGB_PARAMS`, `train_bagged_models`, `prepare_features`.

---

## 4. What to RESEARCH yourself (so you can defend it)

You don't need to invent these methods, but you should understand them well enough
to explain them in the writeup and answer judges' questions. For each: the concept,
the one-line "why," and good search terms.

### 4.1 Causal inference & the do-operator  *(most important)*
- **Concepts:** correlation vs causation; confounders, mediators, colliders;
  Pearl's `do()` operator; Directed Acyclic Graphs (DAGs); Structural Causal
  Models (SCMs); back-door criterion.
- **Why:** This is the heart of Deliverable C and the most-weighted writeup section.
- **Search:** "Judea Pearl do-calculus intuition", "structural causal model tutorial",
  "confounder vs mediator vs collider", "observational vs interventional distribution",
  "DoWhy library tutorial", "back-door criterion explained".
- **Libraries to know exist:** `DoWhy`, `EconML`, `dagitty` (DAG drawing).

### 4.2 Selection bias & inverse propensity weighting
- **Concepts:** Missing-Not-At-Random (MNAR); selection on observables /
  ignorability; positivity/overlap; propensity score; IPW; reject-inference
  (the credit-industry name for exactly this problem).
- **Why:** Deliverable A trains on approved-only data; this de-biases it.
- **Search:** "reject inference credit scoring", "inverse propensity weighting
  explained", "propensity score overlap positivity assumption", "selection bias
  Heckman correction".

### 4.3 Survival analysis (default timing)
- **Concepts:** hazard rate; survival function; censoring; Kaplan-Meier estimator;
  Cox proportional hazards; discrete-time hazard models; cumulative incidence.
- **Why:** Deliverable B is a *timing* problem, not classification.
- **Search:** "survival analysis intuition hazard function", "Kaplan-Meier explained",
  "discrete time survival model", "lifelines python tutorial", "right censoring".
- **Library:** `lifelines`, `scikit-survival`.

### 4.4 Calibration & conformal prediction
- **Concepts:** probability calibration; isotonic vs Platt scaling; reliability
  diagram; conformal prediction; coverage vs interval width; Mondrian / class-
  conditional conformal.
- **Why:** S_cal scores how well your 90% intervals contain truth without being wide.
- **Search:** "probability calibration isotonic regression", "conformal prediction
  tutorial", "Mondrian conformal prediction", "reliability diagram calibration".
- **Library:** `MAPIE` (conformal in scikit-learn style), `sklearn.calibration`.

### 4.5 Gradient boosting & lending economics
- **Concepts:** gradient-boosted decision trees; handling NaN/categoricals;
  Probability of Default (PD), Loss Given Default (LGD), Exposure at Default (EAD);
  expected loss = PD × LGD × EAD; risk-based pricing.
- **Why:** the model engine + the profit math behind Deliverable A.
- **Search:** "gradient boosting intuition", "PD LGD EAD credit risk", "expected
  loss credit", "HistGradientBoosting vs LightGBM".

---

## 5. What YOU must do by hand (👤)

These are not coding tasks — only you can do them:

1. **Register the team** on the Google Form (deadline **8 PM Friday**). No upload
   link without it; no team changes after.
2. **Put your team name** into `submission_D_writeup.md` (replace `<your team name>`).
3. **Export the writeup to PDF** named exactly `submission_D_writeup.pdf`, placed in
   the `submission/` folder. (Options: open the `.md` in VS Code → "Markdown PDF"
   extension; or paste into Google Docs → Export PDF; keep ≥11pt font, ≥0.75in
   margins, ≤4 pages.)
4. **Run the validator** and confirm `PASS`:
   ```bash
   pip install -r requirements.txt
   python solution.py
   python validate_submission.py ./submission
   ```
5. **Upload all four files** to your team's private link.
6. **(Optional but recommended)** Read the §4 research topics so you can defend the
   methodology if judges ask. The writeup is human-graded on substance.

---

## 6. Suggested order of attack (if maximizing score)

1. **Submit-ready first:** confirm validator PASS, fill team name, export PDF. This
   guarantees you have a valid, scoring submission no matter what. *(mostly 👤)*
2. **P1 — SCM for C.** Biggest differentiator. *(Kiro builds, you research §4.1)*
3. **P2 — IPW for A**, kept only if it helps on validation. *(Kiro builds + measures)*
4. **P3 — survival model for B.** *(Kiro builds, you research §4.3)*
5. **P4 — per-bin conformal.** *(Kiro builds)*
6. **P5 — stronger learner + features.** *(Kiro builds)*
7. **Refresh the writeup** with new figures/results, re-export PDF. *(Kiro drafts,
   you export)*
8. **Re-validate → re-upload.** *(👤)*

---

## 7. Honest expectation setting

- The ✅ items already give you a **valid, competitive** submission that passes the
  hard validator gate.
- The recovery-economics fix (#5) was a **large, guaranteed** profit improvement.
- **P1 (causal C)** and **P2 (IPW)** are where "competitive" becomes "winning," but
  P2's *numeric* benefit is uncertain (it depends on hidden info the old lender may
  have used) — so we measure it and keep it only if it helps. Either way it earns
  writeup credit.
- Nobody can promise a literal "100%" because the exact scoring weights are withheld
  and A/B/C are graded against hidden ground truth. The plan above maximizes every
  component you can actually control.
