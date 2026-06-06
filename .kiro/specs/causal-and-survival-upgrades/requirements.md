# Requirements Document

## Introduction

This feature enhances two of the three machine-scored deliverables in the SMB
Underwriting Challenge solution pipeline (`solution.py`): the structural causal
model behind Deliverable C (causal counterfactuals) and the default-timing model
behind Deliverable B (cohort default-timing trajectory). Both changes are score
movers identified in the team's `ACTION_AND_RESEARCH.md` plan as the highest
remaining leverage (P1 causal C, P3 survival B).

For Deliverable C, the current pipeline fits only four structural equations over a
hand-specified DAG and falls back to a single deterministic-ratio edit for every
intervention on a feature outside that DAG. This feature expands the causal graph
so that interventions on more intervenable features propagate through fitted
structural equations to their downstream children before re-scoring.

For Deliverable B, the current pipeline mixes three empirical risk-bucket timing
curves. This feature replaces empirical bucketing with a feature-conditioned
discrete-time hazard (survival) model so each cohort receives its own timing
shape, with intervals that propagate timing-curve uncertainty rather than only
incidence sampling noise.

All changes must preserve the existing validator-passing behavior, output file
contract, runtime budget, and dependency constraints. Changes are confined to
`solution.py` unless a strong, documented reason requires otherwise.

This work explicitly excludes: Deliverable A modeling changes, the Deliverable D
writeup PDF export, team registration, and any stronger learner work such as
LightGBM or feature engineering (tracked separately as P5).

## Glossary

- **Pipeline**: The solution program in `solution.py` that produces the three
  machine-scored submission files.
- **Validator**: The program `validate_submission.py` that gates a submission and
  prints `PASS` or `FAIL`.
- **SCM**: The Structural Causal Model used by Deliverable C, consisting of a
  directed acyclic graph of features and one fitted structural equation per child
  node.
- **DAG**: The directed acyclic graph of parent-to-child causal edges that defines
  the SCM, currently held in the `SCM_EDGES` mapping.
- **Structural_Equation**: A regression model fitted from training data that
  predicts one child feature from its parent features.
- **Intervention**: A `do(feature = value)` operation that sets one feature to a
  fixed value, breaks that feature's incoming edges, and re-simulates the
  feature's descendants.
- **Intervenable_Feature**: A feature marked `intervenable = True` in
  `data_dictionary.csv` (the set listed under "Intervenable Feature Set" below).
- **No_Op_Intervention**: An intervention that sets a feature to its current value
  for an applicant.
- **PD**: Probability of default, a calibrated value in `[0, 1]`.
- **Counterfactual_PD**: The post-intervention PD reported for a Deliverable C
  query.
- **Survival_Model**: The discrete-time hazard model for Deliverable B that
  produces a cumulative default fraction by loan age for a given cohort.
- **Hazard_Rate**: The conditional probability that a loan first defaults during a
  given discrete age interval, given it had not defaulted before that interval.
- **Cohort_Curve**: The 13-point sequence of cumulative default fractions for a
  single origination cohort week across loan ages 1 through 13 weeks.
- **Timing_Shape**: The relative distribution of a cohort's defaults across loan
  ages, independent of the cohort's overall default incidence.
- **Cumulative_Default_Rate**: The predicted cumulative default fraction by day
  `7a` for loan age `a` weeks, in `[0, 1]`.
- **Grid**: The full 13x13 = 169-row table of `(cohort_week, loan_age_weeks)`
  pairs required for Deliverable B.
- **Interval_Bounds**: The 90% lower and upper bounds reported alongside a point
  estimate (`*_lower_90` and `*_upper_90` columns).
- **Conformal_Calibration**: The per-bin (Mondrian-style) additive interval
  widening implemented by `tune_perbin_conformal` and `apply_perbin_conformal`.
- **Runtime_Budget**: A full local run of the pipeline completing in approximately
  1 to 2 minutes.
- **Allowed_Dependencies**: The Python libraries numpy, pandas, and scikit-learn,
  with no OpenMP or GPU dependency.

## Intervenable Feature Set

The features marked `intervenable = True` in `data_dictionary.csv`:
`stated_annual_revenue`, `stated_time_in_business`, `requested_amount`,
`observed_monthly_revenue_avg_3mo`, `observed_revenue_trend_3mo`,
`observed_revenue_volatility`, `observed_cash_balance_p10`,
`observed_overdraft_count_3mo`, `payroll_regularity_score`,
`aggregate_credit_utilization`, `recent_inquiries_count_6mo`,
`existing_debt_obligations`, `owner_personal_credit_band`,
`invoice_payment_delinquency_rate`, `application_channel`,
`multi_lender_inquiry_count_30d`.

## Requirements

### Requirement 1: Expanded structural causal graph for Deliverable C

**User Story:** As the lending team, I want the causal graph to cover more
intervenable features and their downstream children, so that counterfactual
interventions propagate through structural equations instead of collapsing to a
single deterministic edit.

#### Acceptance Criteria

1. THE Pipeline SHALL define an SCM DAG that contains a fitted Structural_Equation
   for at least eight distinct child features, where each such child is drawn from
   the bank-feed, bureau-credit, or application-context feature groups and at least
   one child originates from each of these three groups.
2. WHEN constructing the parent set for a child node, THE Pipeline SHALL include
   only parent features whose names match an entry in `data_dictionary.csv`.
3. IF a configured parent feature for a child node is not defined in
   `data_dictionary.csv`, THEN THE Pipeline SHALL exclude that parent from the
   child's Structural_Equation and continue fitting the remaining defined parents.
4. THE Pipeline SHALL maintain the DAG as acyclic such that a single topological
   ordering covering all child nodes exists.
5. WHEN a Structural_Equation is fitted, THE Pipeline SHALL train the equation
   using only the training rows in which the child feature value is non-null.
6. IF the count of training rows with a non-null value for a child feature is fewer
   than 200, THEN THE Pipeline SHALL omit that child's Structural_Equation from the
   fitted SCM and continue fitting the remaining Structural_Equations.
7. THE Pipeline SHALL fit every Structural_Equation using only Allowed_Dependencies.

### Requirement 2: Intervention propagation through the SCM

**User Story:** As the lending team, I want `do(feature = value)` to ripple through
all downstream children in causal order, so that counterfactual PD reflects the
full structural effect of the intervention rather than a single mediator edit.

#### Acceptance Criteria

1. WHEN an Intervention sets a feature that is a parent of one or more child nodes
   in the DAG, THE Pipeline SHALL re-simulate each affected descendant child, where
   an affected descendant is a fitted child whose parent set intersects the set of
   already-changed features, processing parents before children in topological
   order before re-scoring PD.
2. WHEN re-simulating a child node under an Intervention, THE Pipeline SHALL add
   the applicant's abducted residual for that child, defined as the observed child
   value minus the structural equation prediction on the original (pre-intervention)
   parents, to the structural equation prediction on the post-intervention parents.
3. WHEN a No_Op_Intervention sets a feature to its current value, THE Pipeline SHALL
   leave every child feature value and the resulting PD unchanged.
4. IF an applicant's observed value for a child feature is missing or non-numeric,
   THEN THE Pipeline SHALL leave that child feature unchanged and not propagate a
   re-simulated value to its descendants.
5. WHERE an intervened feature has no fitted child in the DAG, THE Pipeline SHALL
   apply the existing deterministic-mediator edit for that feature.
6. WHEN an Intervention changes the inputs to the engineered ratio
   `requested_amount_to_observed_revenue`, THE Pipeline SHALL rescale the stored
   ratio by the factor `(requested_amount_new / requested_amount_old) *
   (observed_revenue_old / observed_revenue_new)`.
7. IF any input to the engineered ratio rescale is zero, missing, or non-numeric,
   THEN THE Pipeline SHALL omit that input's contribution to the rescale factor
   rather than perform an undefined division.
8. IF an applicant's stored value for the engineered ratio
   `requested_amount_to_observed_revenue` is missing, THEN THE Pipeline SHALL leave
   that ratio unchanged.

### Requirement 3: No-op intervention invariant for Deliverable C

**User Story:** As a regulator-facing analyst, I want intervening a feature to its
current value to leave PD unchanged, so that the causal predictions are internally
consistent and defensible.

#### Acceptance Criteria

1. WHEN a No_Op_Intervention is applied to any intervenable feature for any
   applicant, THE Pipeline SHALL produce a Counterfactual_PD equal to that
   applicant's pre-intervention PD within a tolerance of 1e-6.
2. WHEN a No_Op_Intervention is applied, THE Pipeline SHALL leave every child
   feature value equal to its pre-intervention value within a tolerance of 1e-6.

### Requirement 4: Directional correctness of counterfactual effects

**User Story:** As a regulator-facing analyst, I want interventions to move PD in
the economically correct direction, so that the causal story holds up to scrutiny.

#### Acceptance Criteria

1. WHEN an Intervention increases a feature whose higher values indicate higher
   credit risk, THE Pipeline SHALL produce a Counterfactual_PD that is greater than
   or equal to the applicant's pre-intervention PD, aggregated across applicants.
2. WHEN an Intervention improves a feature whose higher values indicate lower
   credit risk, THE Pipeline SHALL produce a Counterfactual_PD that is less than or
   equal to the applicant's pre-intervention PD, aggregated across applicants.
3. THE Pipeline SHALL report each directional effect aggregate so the team can
   confirm the share of upward, downward, and near-zero movements.

### Requirement 5: Deliverable C output contract preserved

**User Story:** As the submitting team, I want Deliverable C to keep its exact
output schema and coverage, so that the submission continues to pass the Validator.

#### Acceptance Criteria

1. THE Pipeline SHALL write Deliverable C to
   `submission/submission_C_counterfactuals.csv`.
2. THE Pipeline SHALL write one row per `query_id` present in
   `intervention_queries.csv`, covering all expected query identifiers.
3. THE Pipeline SHALL write the columns `query_id`, `predicted_pd_cf`,
   `pd_cf_lower_90`, and `pd_cf_upper_90`.
4. THE Pipeline SHALL produce `predicted_pd_cf`, `pd_cf_lower_90`, and
   `pd_cf_upper_90` values within `[0, 1]` on every row.
5. THE Pipeline SHALL produce values satisfying
   `pd_cf_lower_90 <= predicted_pd_cf <= pd_cf_upper_90` on every row.
6. IF a query references an applicant identifier absent from the scored
   submission set, THEN THE Pipeline SHALL write a valid in-range row for that
   query.
7. THE Pipeline SHALL apply the existing Conformal_Calibration widening to the
   Deliverable C interval bounds.

### Requirement 6: Discrete-time hazard survival model for Deliverable B

**User Story:** As the lending team, I want a feature-conditioned discrete-time
hazard model, so that each cohort gets its own default Timing_Shape rather than a
shared empirical curve.

#### Acceptance Criteria

1. THE Pipeline SHALL fit a Survival_Model that estimates a discrete Hazard_Rate in
   the range `[0, 1]` for each of the 13 loan-age intervals spanning loan age 1
   week through 13 weeks, producing exactly one Hazard_Rate per interval.
2. THE Pipeline SHALL condition the Survival_Model on applicant features so that,
   for any two loans whose conditioned risk differs, the resulting Timing_Shape
   differs in at least one of the 13 loan-age intervals.
3. WHEN the Survival_Model produces a Cohort_Curve, THE Pipeline SHALL compute a
   cumulative default fraction for each loan age 1 through 13 weeks by combining the
   per-interval Hazard_Rates, such that the cumulative default fraction is
   monotonically non-decreasing across loan ages and each value lies in `[0, 1]`.
4. THE Pipeline SHALL derive each of the 13 cohort weeks' Timing_Shape from the
   Survival_Model such that, for any target fraction in `(0, 1]` of a cohort's total
   cumulative defaults, a higher-risk cohort reaches that fraction at a loan age in
   weeks that is less than or equal to the loan age at which a lower-risk cohort
   reaches the same fraction.
5. THE Pipeline SHALL fit the Survival_Model using only Allowed_Dependencies.
6. THE Pipeline SHALL restrict default-timing estimation to the `days_to_default`
   window `[1, 90]` defined in the dataset README, and IF a default record has a
   `days_to_default` value outside `[1, 90]`, THEN THE Pipeline SHALL exclude that
   record from Survival_Model estimation.

### Requirement 7: Timing-uncertainty intervals for Deliverable B

**User Story:** As the submitting team, I want the Deliverable B intervals to
reflect timing-curve uncertainty, so that the 90% bounds capture more than
incidence sampling noise.

#### Acceptance Criteria

1. THE Pipeline SHALL produce 90% Interval_Bounds for each Grid cell that
   incorporate uncertainty in the estimated Timing_Shape.
2. THE Pipeline SHALL produce Interval_Bounds that incorporate uncertainty in each
   cohort's default incidence.
3. THE Pipeline SHALL produce `cdr_lower_90` and `cdr_upper_90` values satisfying
   `cdr_lower_90 <= cumulative_default_rate <= cdr_upper_90` on every Grid cell.

### Requirement 8: Deliverable B output contract and monotonicity preserved

**User Story:** As the submitting team, I want Deliverable B to keep its exact grid
schema and monotonicity guarantee, so that the submission continues to pass the
Validator.

#### Acceptance Criteria

1. THE Pipeline SHALL write Deliverable B to
   `submission/submission_B_trajectory.csv`.
2. THE Pipeline SHALL write all 169 Grid rows covering every
   `(cohort_week, loan_age_weeks)` pair for cohort weeks 1 through 13 and loan ages
   1 through 13.
3. THE Pipeline SHALL write the columns `cohort_week`, `loan_age_weeks`,
   `cumulative_default_rate`, `cdr_lower_90`, and `cdr_upper_90`.
4. THE Pipeline SHALL produce `cumulative_default_rate`, `cdr_lower_90`, and
   `cdr_upper_90` values within `[0, 1]` on every row.
5. WHILE iterating over increasing loan age within a single cohort week, THE
   Pipeline SHALL produce `cumulative_default_rate` values that are non-decreasing.
6. THE Pipeline SHALL write `cohort_week` and `loan_age_weeks` as integers.

### Requirement 9: Submission-wide invariants and operational constraints preserved

**User Story:** As the submitting team, I want the full pipeline to keep passing the
Validator within the existing runtime and dependency limits, so that the upgrades
do not jeopardize a scorable submission.

#### Acceptance Criteria

1. WHEN the Pipeline run completes successfully and writes all three submission
   files, THE Validator SHALL print `PASS` for the `submission/` folder.
2. THE Pipeline SHALL leave the Deliverable A output file
   `submission/submission_A_decisions.csv` unchanged in name and location.
3. THE Pipeline SHALL complete a full local run within the Runtime_Budget.
4. THE Pipeline SHALL run using only Allowed_Dependencies.
5. THE Pipeline SHALL confine all source changes to `solution.py` unless a
   documented reason requires editing another file.
6. THE Pipeline SHALL retain the per-bin Conformal_Calibration behavior for the
   90% intervals across Deliverables A, B, and C.
