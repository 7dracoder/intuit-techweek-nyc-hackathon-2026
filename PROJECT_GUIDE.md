# SMB Underwriting Challenge — Complete Project Guide

> Your one-stop explanation of what this hackathon is, what you must deliver,
> exactly how you're scored, what I've already built, and the precise roadmap to
> a "best-of-the-best" submission.

---

## 1. What this project is

This is the **Intuit TechWeek NYC 2026 "Explainable ML" Hackathon** — an
**SMB (small-business) Underwriting Challenge**.

You role-play a **small-business lender**. You're given a historical book of
~100K loan applications and you must:

1. **Decide whom to fund** to maximize portfolio profit (Deliverable A).
2. **Forecast how the funded loans default over time** (Deliverable B).
3. **Answer causal "what-if" questions** — what happens to risk if we *change*
   one feature (Deliverable C).
4. **Defend your reasoning** in a short technical writeup (Deliverable D).

The word "Explainable" is the whole point: this isn't a pure accuracy contest.
It rewards **calibrated uncertainty**, **genuine causal reasoning**, and a
**defensible methodology** — the kind of thing a real lender could show a
regulator.

### The loan product (the rules of the money)

Every funded loan uses fixed terms:

| Term | Value |
|---|---|
| Amount | `requested_amount` (roughly $5K–$50K) |
| Duration | 60 days, repaid via **daily ACH draws** |
| APR | 35% (annualized) |
| Origination fee | 3% of amount, collected up front |

**Default definition** — a loan defaults if ANY of these happen:
1. **3 consecutive missed draws** (a successful draw resets this counter — a
   borrower can "cure").
2. **6 total missed draws** over the life of the loan (never resets).
3. **Outstanding balance > 0 at day 90** (never paid off in time).

`days_to_default` is the day in [1, 90] it first hit a condition. This definition
is what you model for A (will it default) and B (when it defaults).

---

## 2. The data

After unzipping `dataset/dataset-compressed.zip` you get three tables (44 columns
each):

| File | Rows | What it is |
|---|---|---|
| `train.csv` | 85,340 | History. Outcomes filled in **only** for prior-approved + matured loans. |
| `validation.csv` | 4,489 | Has outcomes — used to tune and calibrate. |
| `test.csv` | 8,817 | **All outcome fields blank** — this is what you're scored on. |

**Critical data facts I confirmed by profiling the real files:**

- In `train.csv`: **51,722 approved** (outcomes present), **33,618 declined**
  (no outcome). Among approved, **~17.4% defaulted**.
- In `validation.csv`: 2,551 labeled rows, ~20.6% default rate.
- **val + test = exactly 13,306** applicant IDs — this is the set you decide on
  for Deliverable A.
- All **300 intervention applicants** (900 queries, 3 per applicant, 30 distinct
  features) live in **test.csv**.
- `days_to_default` spans 3–90 days, **median 37** → there's real timing signal.
- Bank-feed coverage is only **~63%** (37% of rows have null bank-feed columns).
- **The big one:** `final_recovered_amount` on a default averages only ~9% of
  principal — but defaulters repay via daily draws *before* defaulting, so true
  effective recovery is **~70%**. (More on why this matters in Section 5.)

### Feature groups

`business_identity`, `self_reported` (optimistically biased), `bank_feed`
(partial coverage), `bureau_credit`, `platform_engagement`,
`application_context`, `prior_underwriter` (the old lender's score/decision), and
`outcome` (approved-only, blank in test).

The `data_dictionary.csv` has an **`intervenable`** column telling you which
features are valid intervention targets for Deliverable C.

### Assumptions the data deliberately breaks

This is what the writeup wants you to engage with:

- **Selection bias / outcomes Missing-Not-At-Random** — you only see repayment
  for loans a prior underwriter chose to approve. Your training labels are a
  censored, non-random slice of reality.
- **Optimistic self-reported fields** — applicants inflate stated revenue, etc.
- **Partial feature coverage** — bank-feed nulls are not random (only linked
  feeds have them).
- **Right-censoring** — immature loans have no final outcome yet.

---

## 3. What you must submit — the four deliverables

You upload **exactly four files**, with **exactly these names**, flat in one
folder (no subfolders). A wrong name or missing ID means it cannot be scored.

### A — `submission_A_decisions.csv` (your lending policy)
- One row per applicant in validation + test = **13,306 rows**.
- Columns: `applicant_id`, `decision` (1=approve / 0=decline),
  `predicted_pd` (PD in [0,1], **required even for declines**),
  `pd_lower_90`, `pd_upper_90`.
- Rule: `pd_lower_90 <= predicted_pd <= pd_upper_90` on every row.

### B — `submission_B_trajectory.csv` (default-timing forecast)
- The full 13×13 = **169-row grid** (cohort_week 1–13 × loan_age_weeks 1–13).
- For each cohort `w` and age `a`, predict the cumulative fraction of **your
  approved cohort-w loans** that defaulted by day `7a`.
- Columns: `cohort_week`, `loan_age_weeks`, `cumulative_default_rate`,
  `cdr_lower_90`, `cdr_upper_90`.
- Rules: interval ordering, AND **non-decreasing in age within each cohort**
  (cumulative rates can only go up). Use `submission_B_template.csv` — overwrite
  only the three prediction columns; never change the grid.

### C — `submission_C_counterfactuals.csv` (causal what-if)
- One row per `query_id` in `intervention_queries.csv` = **900 rows**.
- Each asks: what's this applicant's PD **if we `do(feature = value)`** — set one
  feature by intervention, hold everything else fixed.
- Columns: `query_id`, `predicted_pd_cf`, `pd_cf_lower_90`, `pd_cf_upper_90`.

### D — `submission_D_writeup.pdf` (technical writeup)
- Max **4 pages** body, **≥11pt** font, **≥0.75in** margins.
- Five fixed sections, in order:
  1. Problem framing & assumptions violated
  2. Methodology
  3. **Causal reasoning & counterfactual methodology** ← weighted most heavily
  4. Calibration & uncertainty quantification
  5. Limitations & what we'd do differently

---

## 4. How you're scored (from the brief)

The A/B/C files are scored automatically; D is human-reviewed. Final score
combines:

| Symbol | What it measures | Driven by |
|---|---|---|
| **S_P&L** | Realized portfolio profit of the loans you funded | A decisions |
| **S_traj** | Accuracy of your cohort default-timing forecast | B |
| **S_cal** | Whether your 90% intervals contain the truth *without being needlessly wide* | A and B |
| **S_C** | Closeness of your interventional PDs to true causal effects (explicitly **not** naive re-prediction) | C |
| **S_write** | Quality of your methodological defense | D |

The exact weights/formulas are **deliberately withheld**. A "winning team":
funds a genuinely profitable book, accurately forecasts default timing, gives
well-calibrated uncertainty, returns true interventional CFs, and defends every
choice clearly.

---

## 5. What you (the human) actually need to do

A realistic checklist, in order:

1. **Register the team** on the Google Form (deadline **8 PM Friday**). You don't
   get an upload link until you register. No team changes after.
2. **Unzip the dataset** — done (the CSVs are now extracted in `dataset/`).
3. **Build the four submission files** — the A/B/C generator is done (see
   Section 6). For D, export the writeup to PDF.
4. **Validate** until it prints `PASS`:
   ```bash
   pip install -r requirements.txt
   python validate_submission.py ./submission
   ```
5. **Export D to PDF** named exactly `submission_D_writeup.pdf` and drop it into
   the `submission/` folder next to the three CSVs.
6. **Upload all four files** to your team's private link.

> **Hard gate:** `validate_submission.py` must print **PASS**. It checks exact
> names, ID coverage (13,306 / 900 / 169), [0,1] ranges, interval ordering, and
> B monotonicity. A failing submission is **disqualified**.

---

## 6. What I've already built and verified

### Files I created / changed

| File | Status | What it is |
|---|---|---|
| `solution.py` | **new** | Full pipeline → produces A, B, C into `./submission/`. |
| `submission/submission_A_decisions.csv` | **generated** | 13,306 rows, profit-optimal decisions + calibrated PD + 90% PI. |
| `submission/submission_B_trajectory.csv` | **generated** | 169-row monotone cohort×age trajectory + PI. |
| `submission/submission_C_counterfactuals.csv` | **generated** | 900 `do(feature=value)` interventional PDs + PI. |
| `submission_D_writeup.md` | **new** | Filled writeup (export this to PDF). |
| `submission_D_writeup_template.md` | restored | Left untouched (original template). |
| `requirements.txt` | edited | Added `scikit-learn`. |
| `PROJECT_GUIDE.md` | **new** | This document. |

### How `solution.py` approaches each deliverable

- **A — PD + decision.** A bagged ensemble (8 members, bootstrap rows + distinct
  seeds) of scikit-learn's `HistGradientBoostingClassifier`, trained on rows with
  an observed outcome. It handles NaN and integer categoricals natively (no
  leakage-prone imputation). PD is **isotonic-calibrated** on validation. The
  approve/decline decision **maximizes expected profit per loan** — approve iff
  `(1-PD)·income > PD·loss` — not a fixed 0.5 threshold.
- **B — timing.** `CDR(w,a) = cohort_incidence(w) × F(a)`, where `F(a)` is the
  empirical fraction of eventual defaults occurring by week `a` (from
  `days_to_default`), forced non-decreasing. This guarantees monotone curves and
  models *timing*, not a flat number.
- **C — counterfactuals.** Copy the applicant row, **set** the intervened
  feature, **propagate** the change to the deterministic engineered mediator
  (`requested_amount_to_observed_revenue`), and re-score. A structural edit, not a
  blind re-prediction.
- **Uncertainty (A/B/C).** Intervals from ensemble 5th/95th percentiles, widened
  by an additive **conformal** term tuned on validation for ~90% coverage, then
  clamped to [0,1] and ordered.

### Measured quality (on labeled validation)

- **AUC ≈ 0.7544** unweighted / **0.7525** final (IPW kept), **Brier ≈ 0.134**.
- **40 features (5 engineered)** — revenue consistency, debt-service coverage,
  utilization×inquiries, cash-to-requested, prior-default ratio.
- **Calibration is near-perfect** — predicted vs. observed default rate match
  across deciles (see Figure 2 in the writeup).
- **100% bin-wise interval coverage** at mean width 0.232 (Figure 3).
- Decision policy approves the profitable low-PD book; profit-simulated threshold
  ≈ 0.21. Approval rate **~64%**.
- SCM (Deliverable C): 10 fitted equations; no-op interventions move PD by exactly
  0.000 (asserted in `pytest`); 6/6 tests pass.
- Validator: **PASS** (0 errors; only the expected "writeup PDF not found"
  warning until you export the PDF).

### The one important bug I caught and fixed

My first run approved only **12.8%** of applicants because I estimated recovery
from `final_recovered_amount` alone (~9% of principal). But a defaulted loan
amortizes via **daily ACH draws** until it defaults (median day 37 of 60), and
`final_recovered_amount` is only the *additional* post-default recovery. Counting
both, effective recovery is **~70%** (loss-given-default ~30%, not ~91%).
Correcting this moved approval to a sensible **~67%** — a direct, large
improvement to the portfolio-profit score (S_P&L).

---

## 7. Roadmap to the best-of-the-best (100%) submission

The current pipeline is a strong, defensible, validator-passing baseline. Here is
the prioritized work to push each scored component toward the top. I have **not**
done these yet — they're the plan.

### Priority 1 — Deliverable C: a real Structural Causal Model (highest leverage)
`S_C` explicitly penalizes "naive re-prediction," and Section 3 of the writeup is
weighted most. Right now we only propagate to one *deterministic* mediator.
**Plan:**
- Build a defended **DAG** over the features using the `intervenable` flags and
  credit-domain reasoning (e.g. revenue → cash balance → overdrafts; utilization
  → default; requested_amount → amount/revenue ratio).
- Fit lightweight structural equations (each child regressed on its parents).
- For `do(feature=v)`: set the node, **re-simulate its non-deterministic
  children** through the structural equations, then score PD. This captures
  indirect interventional effects we currently miss.
- This is the single biggest differentiator vs. other teams.

### Priority 2 — Deliverable A: de-bias the selection / sharpen profit
- Add a **selection model / inverse-propensity weighting** (or a two-stage
  accept-then-default model) so PD generalizes to the never-funded region, not
  just the prior lender's approved slice.
- Replace the portfolio-average recovery with a **per-loan LGD model** (predict
  recovery from features) — this matters most for applicants near the break-even
  PD, where approval decisions flip.
- Tune the profit threshold with a small **portfolio simulation** on validation
  (sum realized profit under different cutoffs) rather than the analytic break-even.

### Priority 3 — Deliverable B: feature-conditioned survival model
- Replace the single global timing curve `F(a)` with a **discrete-time hazard
  model** (or Cox / `lifelines`) conditioned on features, so each cohort gets its
  own realistic shape (cohorts differ in risk mix).
- Derive B intervals from the survival model's variance + cohort bootstrap rather
  than incidence bootstrap alone — tighter and better-calibrated → higher S_cal.

### Priority 4 — Calibration polish (S_cal)
- Move from a single additive conformal delta to **per-risk-bin / Mondrian
  conformal** widening so coverage is right *across the PD range*, not just on
  average. Narrower intervals at the same coverage = better S_cal.
- Cross-validated calibration instead of a single validation split.

### Priority 5 — Modeling strength (S_P&L, S_traj)
- Install LightGBM/XGBoost (needs `libomp` on macOS:
  `brew install libomp`) for a stronger learner, and **stack** it with the
  HistGB ensemble.
- Light feature engineering: revenue-consistency (stated vs. observed), debt
  service coverage, prior-default ratio, bank-feed-linked flag interactions.
- Hyperparameter search (Optuna) with time-aware CV.

### Priority 6 — Deliverable D: make the writeup airtight
- Add the DAG figure and an explicit observational-vs-interventional example.
- Show a calibration plot and coverage table.
- Keep it to 4 pages, ≥11pt, ≥0.75in margins, five sections in order, then
  export to `submission_D_writeup.pdf`.

### Definition of "done / best"
- Validator prints **PASS**.
- Validation AUC pushed toward ~0.78+ with maintained calibration.
- Profit-simulated decision threshold (not just analytic).
- C uses a real SCM with non-deterministic propagation.
- B uses a feature-conditioned hazard model.
- Intervals are bin-wise ~90% covered and as tight as possible.
- D is a complete, figure-backed, regulator-ready 4-pager.

---

## 8. Quick command reference

```bash
# 1. install deps
pip install -r requirements.txt

# 2. (one-time) unzip data — already done
cd dataset && unzip -o dataset-compressed.zip && cd ..

# 3. generate A/B/C
python solution.py

# 4. validate (must print PASS)
python validate_submission.py ./submission

# 5. export submission_D_writeup.md -> submission/submission_D_writeup.pdf
#    then upload all four files to your team's private link
```

---

*Note: the writeup currently lists `<your team name>` as a placeholder — replace
it with your real team name before exporting to PDF.*
