# What We Built — Reference for the Writeup

> A plain-English record of every technical decision made for the causal and
> survival upgrades. Use this to fill in Deliverable D (submission_D_writeup.md).

---

## High-level story

The starting pipeline already passed the validator and produced all three
submission files. The two remaining gaps — flagged in ACTION_AND_RESEARCH.md as
the highest-remaining score movers — were:

1. **Deliverable C (causal counterfactuals):** interventions only edited a single
   deterministic ratio; they did not propagate to downstream features. Riskier or
   safer counterfactual scenarios were therefore underestimated.
2. **Deliverable B (default-timing trajectory):** all cohorts shared the same
   empirical timing curve, so a riskier cohort looked identical in timing shape
   to a safer one.

Both were fixed inside `solution.py` without touching Deliverable A, the output
file names/locations, or the dependency set (numpy / pandas / scikit-learn only).

---

## Enhancement 1 — Deeper Structural Causal Model (Deliverable C)

### What the problem was

The original SCM (`SCM_EDGES`) fitted only **four** structural equations. Every
intervention on a feature outside those four fell back to editing a single
deterministic ratio (`requested_amount_to_observed_revenue`). This means that
intervening on, say, revenue did not change downstream cash balance, overdrafts,
or credit utilization — the model was blind to those causal pathways.

### How we fixed it

We expanded `SCM_EDGES` from 4 to **10 fitted structural equations** grouped into:

- **Bank-feed group (6 children):**
  `observed_monthly_revenue_avg_3mo`, `observed_revenue_volatility`,
  `observed_revenue_trend_3mo`, `observed_cash_balance_p10`,
  `payroll_regularity_score`, `observed_overdraft_count_3mo`

- **Bureau-credit group (3 children):**
  `existing_debt_obligations`, `recent_inquiries_count_6mo`,
  `aggregate_credit_utilization`

- **Application-context group (1 child):**
  `multi_lender_inquiry_count_30d`

The causal DAG flows:
```
(stated_annual_revenue, sector, employee_count_bucket, vintage_years,
 stated_time_in_business)
    → MR (monthly revenue avg)
         → VOL (revenue volatility)
              → TREND, PAY (payroll regularity), CASH (cash balance p10)
                   → OD (overdraft count)
(existing_debt_obligations) → CASH, UTIL
(multi_lender_inquiry_count_30d) → INQ (recent inquiries 6mo) → UTIL
```

No back-edges exist, so the graph is acyclic. A single topological ordering covers
all 10 children. Every parent name is a verified `data_dictionary.csv` column.

### The algorithm (unchanged from before)

The `scm_intervene` function implements Pearl's **abduction–action–prediction**
three-step do-operator:

1. **Abduct** — for each fitted child, record the applicant's residual:
   `u = observed_child − regressor.predict(original_parents)`. This captures
   unobserved idiosyncratic variation for that specific applicant.
2. **Act** — force `row[feature] = value` (breaking that feature's incoming edges).
3. **Predict** — in topological order, re-simulate every downstream child whose
   parent set intersects the changed-feature set:
   `new_child = regressor.predict(new_parents) + u`.
   Then re-score PD with the ensemble.

The **no-op invariant** holds by construction: when `value == current_value`, step
3 produces `regressor.predict(same_parents) + u = original_child` for every child,
so PD is unchanged within 1e-6.

### Robustness guards (already in the code, verified for 10 children)

- A child with fewer than 200 non-null training rows is skipped (no equation fitted).
- An all-NaN parent column does not crash; `HistGradientBoostingRegressor` handles
  NaN natively.
- A child with a missing/non-numeric observed value is left unchanged and not
  propagated to its descendants.
- The deterministic engineered ratio `requested_amount_to_observed_revenue` is
  rescaled by `(amt_new/amt_old) × (rev_old/rev_new)` with zero/NaN guards.

### New diagnostic

`report_directional_effects` — a print-only helper called inside
`build_deliverable_C` that tallies the share of upward, downward, and near-zero
counterfactual PD movements per query run. It writes nothing to the submission;
it lets the team confirm directional correctness at a glance.

Observed at runtime: **≈16% up / 84% down / 0% near-zero** across the 900
queries — directionally correct, since the query set is dominated by
improvement-type interventions, so most counterfactual PDs go down.

---

## Enhancement 2 — Discrete-time Hazard Survival Model (Deliverable B)

### What the problem was

The original `estimate_timing_by_risk` divided loans into three PD buckets and
computed one empirical timing curve per bucket. All cohorts in the same bucket got
the same timing shape regardless of their specific feature mix. Riskier cohorts
did not default faster than safer ones in shape — only in total count.

### How we fixed it

We replaced empirical bucketing with a **feature-conditioned discrete-time hazard
model**, following the Singer & Willett "person-period" approach. This is the
natural choice given the constraint to scikit-learn only: it turns survival
estimation into a standard binary classification on expanded rows.

#### Step 1: Person-period expansion (`expand_person_periods`)

Each labeled training loan is expanded into up to 13 discrete-time rows, one per
weekly age interval:

- **Defaulters** with `days_to_default ∈ [1, 90]` → emit rows for ages
  `a = 1 … a*`, where `a* = min(13, ceil(days_to_default / 7))`. The event
  indicator is `0` for `a < a*` and `1` at `a = a*`. Rows after `a*` are omitted
  (the loan has left the risk set).
- **Non-defaulters** (matured, `default_flag == 0`) → emit 13 rows, all
  `event = 0`.
- Records with `days_to_default` outside `[1, 90]` or NaN are excluded from
  defaulter timing (per dataset README constraint).

Each row carries the loan's full feature vector plus `loan_age_weeks` (= `a`) as
an integer covariate.

#### Step 2: Bagged hazard model fitting (`fit_hazard_models`)

Fits `N_BAG_SURV = 8` `HistGradientBoostingClassifier` models on bootstrapped
person-period rows. Each model predicts the **conditional hazard**:

```
h(a | x) = P(event = 1 | features = x, loan_age = a)
```

`loan_age_weeks` is treated as numeric; other categoricals use the existing mask.

#### Step 3: Applicant cumulative curve (`applicant_cumulative_curve`)

For each applicant, build 13 covariate rows (ages 1–13), predict the hazard
vector from the model, then:

```
S(a) = ∏_{k=1..a} (1 − h(k))   # survival function
F(a) = 1 − S(a)                  # cumulative default fraction
```

`F` is guaranteed non-decreasing and in `[0, 1]` because each `h(k) ∈ [0, 1]`.

#### Step 4: Cohort aggregation (`aggregate_cohort_curves`)

For each cohort week `w`, average the incidence-scaled curve `G(a)` over the
approved applicants in that cohort (using `assign_cohort_week`). Empty cohorts
fall back to the mean curve over all approved applicants.

**Level calibration (important fix):** The hazard classifier is well-calibrated
in *shape* (the relative distribution of defaults across loan age) but
over-predicts the *level* by ~5x — a powerful gradient-booster on person-period
rows learns each defaulter's feature signature and assigns elevated hazard to all
of that loan's weekly rows. Raw `F(13)` therefore lands near 0.50 instead of the
true ~0.17 default rate. We fix this by decoupling shape from level
(`scale_curve_to_incidence`):

```
shape_i(a) = F_i(a) / F_i(13)        # normalized timing curve, ends at 1.0
G_i(a)     = PD_i × shape_i(a)        # scaled by calibrated incidence
```

So each applicant's terminal cumulative default fraction equals their calibrated
PD (the well-calibrated level from the Deliverable A model), while the timing
*shape* still comes from the feature-conditioned survival model. Riskier cohorts
reach a higher terminal level AND can default at a different pace.

#### Step 5: Uncertainty intervals (`survival_intervals`)

For `N_BOOT_SURV = 200` bootstrap iterations, **three** sources of uncertainty
are propagated simultaneously:
- **Timing-shape uncertainty:** pick one of the 8 bagged hazard models.
- **Incidence sampling uncertainty:** resample the cohort's approved applicants
  with replacement.
- **PD-level (model) uncertainty:** apply a single shared per-iteration shift of
  every applicant's incidence within its own 90% PD band (from Deliverable A).
  Because the shift is shared across applicants it is *systematic* (correlated),
  so it does NOT average away over a cohort — this is the dominant source of
  trajectory-level uncertainty.

The 5th/95th percentiles become `cdr_lower_90` / `cdr_upper_90`. This produces
honest bands (mean width ~0.10) that widen with loan age as cumulative
uncertainty compounds, rather than the unrealistically tight bands (~0.006) you
get from shape+incidence resampling alone.

#### Output contract (preserved)

- 169 rows covering all (cohort_week, loan_age_weeks) pairs for weeks 1–13 × ages
  1–13.
- `cumulative_default_rate` is non-decreasing within each cohort
  (`np.maximum.accumulate` applied per cohort on point and bounds before writing).
- All values clamped to `[0, 1]` with `cdr_lower_90 ≤ cumulative_default_rate ≤
  cdr_upper_90`.
- `cohort_week` and `loan_age_weeks` written as integers.

---

## Constants added to `solution.py`

```python
N_BAG_SURV = 8       # bagged hazard models
N_BOOT_SURV = 200    # bootstrap resamples for intervals
N_AGE_WEEKS = 13     # weekly loan-age intervals
DTD_MIN, DTD_MAX = 1, 90   # days_to_default window
```

---

## What was NOT changed

- `build_deliverable_A` — left completely untouched.
- Output file names and locations (`submission_A/B/C_*.csv`).
- The per-bin (Mondrian) conformal calibration — still applied to all three
  deliverables' interval bounds.
- The dependency set — strictly numpy, pandas, scikit-learn; no LightGBM, no
  external causal libraries.
- `validate_submission.py` — still prints **PASS** after the upgrades.

---

## Functions added / modified

| Function | File | Action |
|---|---|---|
| `SCM_EDGES` | `solution.py` | Expanded from 4 to 10 children |
| `expand_person_periods` | `solution.py` | New — person-period row expansion |
| `fit_hazard_models` | `solution.py` | New — bagged discrete-time hazard |
| `applicant_cumulative_curve` | `solution.py` | New — hazards → F(a) per applicant |
| `scale_curve_to_incidence` | `solution.py` | New — decouple timing shape from level; G(a) = PD × shape(a) |
| `aggregate_cohort_curves` | `solution.py` | New — mean G(a) per cohort week |
| `survival_intervals` | `solution.py` | New — 3-source bootstrap 90% bounds (shape + incidence + PD-level) |
| `build_deliverable_B` | `solution.py` | Reworked body (same signature) |
| `report_directional_effects` | `solution.py` | New — print-only diagnostic |
| `build_deliverable_C` | `solution.py` | Calls directional diagnostic (same output) |
| `fit_scm`, `scm_intervene`, `_topo_order`, `_rescale_ratio` | `solution.py` | Unchanged — automatically benefit from wider SCM_EDGES |
| `test_solution.py` | repo root | New — test scaffolding with session fixtures |
| `N_BAG_SURV`, `N_BOOT_SURV`, `N_AGE_WEEKS`, `DTD_MIN/MAX` | `solution.py` | New constants |

---

## Key concepts to reference in the writeup

### For Deliverable C (causal section — highest-weighted)

- **Structural Causal Model (SCM):** a DAG where each node is a feature and each
  directed edge represents a causal mechanism, fitted as a regression from data.
- **do(X = v) operator (Pearl):** sets X to v, breaks X's incoming edges (severs
  causes of X), and re-simulates X's effects downstream.
- **Abduction–action–prediction:** the three-step procedure that reuses the
  applicant's idiosyncratic residuals so interventions preserve their individuality
  rather than replacing them with an "average" applicant.
- **Topological order:** ensures parents are re-simulated before their children —
  the causal propagation is computed in the right order.
- **No-op invariant:** intervening a feature to its current value leaves PD exactly
  unchanged. This is a key correctness guarantee for regulators.

### For Deliverable B (survival section)

- **Discrete-time hazard model (Singer & Willett):** turns a survival problem into
  binary classification on person-period rows; the hazard `h(a|x)` is the
  probability of first default in week `a` given survival to that week.
- **Person-period expansion:** each loan contributes one row per weekly interval
  up to its event time; this is the standard setup for discrete-time survival.
- **Survival function S(a) = ∏(1 − h(k)):** product formula over hazards gives
  the probability of surviving to age `a`.
- **Cumulative default fraction F(a) = 1 − S(a):** what the submission actually
  reports — guaranteed monotone by the product form.
- **Bagged ensemble:** reuse of the same `HistGradientBoostingClassifier` already
  powering Deliverable A; no new dependency.
- **Three-source intervals:** the B bootstrap propagates timing-shape uncertainty
  (bagged hazard model selection), incidence sampling uncertainty (cohort applicant
  resampling), and **systematic PD-level uncertainty** (shared per-iteration shift
  within each applicant's 90% PD band). The level term is correlated across
  applicants and survives cohort averaging — it is the dominant source and
  produces honest ~0.10-wide bands that grow with loan age.

---

## Verified results (final, post all fixes)

- Validator: **PASS** (0 errors, 1 warning — missing PDF writeup, human step)
- Validation AUC: **0.7518** (unchanged — A is untouched)
- Brier score: **0.1338**
- Per-bin conformal deltas: `[0. 0. 0.082 0.072 0.046 0.05 0.05 0.072 0. 0.]`
- Approval rate: **63.6%** of 13,306 applicants
- Mean effective recovery (LGD): **0.697**
- Profit-simulated PD threshold: **0.230**
- Deliverable B: **169-row grid, 0 monotonicity violations, 0 interval violations**
  - Terminal CDR per cohort: **~0.134–0.141** (matches approved cohort mean PD, corrected from broken ~0.50)
  - Mean B interval width: **~0.10** (corrected from unrealistic 0.006; propagates 3 uncertainty sources)
  - Intervals widen with loan age: ~0.014 at age 1, ~0.19 at age 13
  - 8 hazard models on **613,893 person-period rows**
- Deliverable C: **900 rows written** — 10 structural equations fitted
- SCM directional effects: **15.7% up / 84.2% down / 0.1% near-zero**
  (economically correct — query set dominated by improvement interventions)

## Bug fixed post-deployment

**Deliverable B level calibration.** The survival model's raw `F(13)` was ~0.50
for every cohort instead of the true ~0.14. Root cause: `HistGradientBoostingClassifier`
on person-period rows learns each defaulter's feature signature and assigns high
hazard to all their weekly rows, inflating the level ~5x while the shape remains
correct. Fix: `scale_curve_to_incidence` normalizes each applicant's curve to
end at 1.0 (pure shape), then multiplies by their calibrated PD. Terminal CDR
now equals the cohort's mean calibrated PD.

**Deliverable B interval width.** After the level fix, the intervals were still
only ~0.006 wide — capturing only timing-shape and applicant-resampling noise, 
which nearly cancels out over a cohort. The dominant source (systematic PD-model
error, correlated across applicants) was missing. Fix: `survival_intervals` now
draws a shared per-iteration standard-normal shift scaled by each applicant's PD
band width. Width grew to ~0.10 and properly widens with loan age.
