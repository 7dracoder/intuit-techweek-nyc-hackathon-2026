# Implementation Plan: Causal and Survival Upgrades

## Overview

This plan implements two enhancements inside `solution.py`, in Python using only
numpy/pandas/scikit-learn (plus Hypothesis for tests):

- **Enhancement 1 (Deliverable C):** Expand `SCM_EDGES` from 4 to 10 fitted
  structural equations across bank-feed, bureau-credit, and application-context
  groups, reusing the existing `fit_scm` / `scm_intervene` / `_topo_order` /
  `_rescale_ratio` machinery and adding a directional-effects diagnostic.
- **Enhancement 2 (Deliverable B):** Replace the empirical risk-bucket timing
  curves with a feature-conditioned discrete-time hazard (survival) model built
  from person-period expansion and bagged `HistGradientBoostingClassifier`s, with
  intervals that propagate both timing-shape and incidence uncertainty.

Tasks are sequenced so each builds on the last and ends wired into `main()` with no
orphaned code. Sub-tasks marked `*` are optional tests and can be skipped for a
faster MVP. Property tests reference properties from the design's "Correctness
Properties" section. All work stays in `solution.py`, with tests in
`test_solution.py`.

## Tasks

- [x] 1. Set up test scaffolding and shared survival/SCM constants
  - Create `test_solution.py` importing `solution` and `hypothesis`, with a small
    cached subsample fixture of training rows for fast model-dependent tests
  - Add `hypothesis` to test tooling notes (do not add it as a `solution.py` import)
  - Add new module-level constants in `solution.py`: `N_BAG_SURV = 8`,
    `N_BOOT_SURV = 200`, `N_AGE_WEEKS = 13`, `DTD_MIN, DTD_MAX = 1, 90`
  - _Requirements: 9.4, 9.5, 1.7, 6.5_

- [x] 2. Expand the SCM DAG configuration (Enhancement 1)
  - [x] 2.1 Rewrite the `SCM_EDGES` constant to the 10-child DAG
    - Encode the 6 bank-feed children, 3 bureau-credit children, and 1
      application-context child with their parent lists exactly as in the design's
      "Proposed SCM DAG Specification"
    - Use only `data_dictionary.csv` column names as parents; keep the graph acyclic
      per the documented topological order
    - _Requirements: 1.1, 1.2, 1.4_

  - [ ]* 2.2 Write property test for DAG acyclicity and topological ordering
    - **Property 2: DAG acyclicity and topological ordering**
    - **Validates: Requirements 1.4, 2.1**
    - Generate random acyclic edge subsets plus the real `SCM_EDGES`; assert
      `_topo_order` lists each child once with parents before children; assert a
      generated cyclic graph is detected/rejected

  - [ ]* 2.3 Write smoke test for SCM structure
    - Assert fitted SCM has >= 8 children, >= 1 per group (bank-feed,
      bureau-credit, application-context), and every parent is a
      `data_dictionary.csv` field
    - _Requirements: 1.1, 1.2_

- [x] 3. Verify and harden SCM fitting over the expanded graph (Enhancement 1)
  - [x] 3.1 Confirm `fit_scm` fits all configured children with guards
    - Ensure each child is fit on rows where the child value is non-null and that
      children with fewer than 200 non-null rows are omitted; ensure an
      undefined/all-NaN parent is tolerated rather than crashing
    - Adjust `fit_scm` only if the expanded graph surfaces a gap; keep the signature
    - _Requirements: 1.3, 1.5, 1.6, 1.7_

  - [ ]* 3.2 Write edge-case tests for SCM fitting guards
    - Assert a thin child (<200 non-null rows) is omitted; assert a bogus parent
      name is tolerated; assert remaining equations still fit
    - _Requirements: 1.3, 1.6_

- [x] 4. Verify intervention propagation through the expanded SCM (Enhancement 1)
  - [x] 4.1 Confirm `scm_intervene` propagates over all fitted children
    - Verify abduct-act-predict uses each child's residual
      `actual - reg.predict(original_parents)` added to
      `reg.predict(new_parents)`, re-simulating descendants in `_topo_order` whose
      parent set intersects the changed-feature set; verify missing/non-numeric
      observed child values are left unchanged and not propagated; verify the
      deterministic-ratio fallback runs for non-SCM features
    - Adjust `scm_intervene` only if the expansion surfaces a gap; keep the signature
    - _Requirements: 2.1, 2.2, 2.4, 2.5_

  - [ ]* 4.2 Write property test for engineered-ratio rescale correctness
    - **Property 3: Engineered-ratio rescale correctness**
    - **Validates: Requirements 2.6**
    - Generate random `r_old` and strictly positive `amt/rev` quadruples; assert
      `_rescale_ratio` returns `r_old * (amt_new/amt_old) * (rev_old/rev_new)` and
      equals `r_old` at equal inputs

  - [ ]* 4.3 Write edge-case tests for ratio guards and non-SCM intervention
    - Assert zero/missing/non-numeric ratio inputs are omitted from the factor;
      assert a NaN stored ratio is unchanged; assert intervening a non-SCM feature
      uses only the deterministic-ratio path
    - _Requirements: 2.5, 2.7, 2.8_

  - [ ]* 4.4 Write property test for the no-op intervention invariant
    - **Property 1: No-op intervention invariance**
    - **Validates: Requirements 2.3, 3.1, 3.2**
    - Generate random applicant index + random intervenable feature; apply
      `do(feature = current_value)`; assert `|pd_cf - pd_base| <= 1e-6` and every
      fitted child delta `<= 1e-6` (run against a small fitted SCM on a subsample)

- [x] 5. Checkpoint - SCM expansion stable
  - Ensure all tests pass, ask the user if questions arise.

- [x] 6. Add directional-effects diagnostic and wire into Deliverable C (Enhancement 1)
  - [x] 6.1 Implement `report_directional_effects` and call it in `build_deliverable_C`
    - Add a print-only helper that, per query, compares `predicted_pd_cf` to the
      applicant baseline PD and tallies upward / downward / near-zero
      (`|delta| <= 1e-6`) shares; invoke it inside `build_deliverable_C` without
      changing the written output
    - Keep the existing unknown-applicant fallback row, ensemble + isotonic scoring,
      per-bin conformal widening, `clamp_intervals`, and four-column CSV write
    - _Requirements: 4.3, 5.1, 5.2, 5.3, 5.6, 5.7_

  - [ ]* 6.2 Write property test for directional correctness of counterfactual effects
    - **Property 4: Directional correctness of counterfactual effects**
    - **Validates: Requirements 4.1, 4.2**
    - For a risk-signed intervenable feature, intervene worse-than-current and
      better-than-current across a random applicant subset; assert mean
      counterfactual PD is `>=` (worse) / `<=` (better) than mean baseline PD,
      aggregated with small tolerance

  - [ ]* 6.3 Write property tests for Deliverable C interval ordering and bounds
    - **Property 8: Interval ordering after calibration**
    - **Validates: Requirements 5.5, 7.3**
    - **Property 9: Output values within the unit interval**
    - **Validates: Requirements 5.4, 8.4**
    - Generate random `(point, lo, hi)` triples (including inverted/out-of-range);
      assert `clamp_intervals` output satisfies `lo <= point <= hi` and all three in
      `[0, 1]`

- [x] 7. Implement person-period expansion for the survival model (Enhancement 2)
  - [x] 7.1 Implement `expand_person_periods(labeled, feature_cols)`
    - Expand each labeled loan into discrete-time rows over 13 weekly intervals:
      defaulters emit rows `a = 1..a*` with `event=1` at
      `a* = min(13, ceil(days_to_default/7))`; matured non-defaulters emit 13 rows
      all `event=0`; carry the `loan_age_weeks` integer age covariate
    - Exclude records whose `days_to_default` is outside `[1, 90]` and defaulters
      with NaN `days_to_default` from defaulter timing
    - Return stacked `(X_pp, y_pp, age_pp)`
    - _Requirements: 6.6_

  - [ ]* 7.2 Write edge-case tests for person-period expansion windowing
    - Assert `days_to_default` outside `[1, 90]` is excluded; assert event interval
      mapping `a* = min(13, ceil(d/7))` and row counts per loan type
    - _Requirements: 6.6_

- [ ] 8. Implement hazard model fitting and applicant curves (Enhancement 2)
  - [ ] 8.1 Implement `fit_hazard_models(X_pp, y_pp, categorical_cols, n_bag)`
    - Fit `N_BAG_SURV` bagged `HistGradientBoostingClassifier`s on bootstrapped
      person-period rows; treat the age covariate as numeric and reuse the existing
      categorical mask; each model predicts `h(a | x)` in `[0, 1]`
    - _Requirements: 6.1, 6.2, 6.5_

  - [x] 8.2 Implement `applicant_cumulative_curve(model, X_app, categorical_cols)`
    - For each applicant build 13 age rows, predict the hazard vector, clip hazards
      to `[0, 1]`, compute `S(a) = prod (1 - h(k))` and `F(a) = 1 - S(a)`
    - _Requirements: 6.1, 6.3_

  - [ ]* 8.3 Write property test for hazard vector validity
    - **Property 5: Hazard vector validity**
    - **Validates: Requirements 6.1**
    - Generate random feature rows; assert the model yields exactly 13 hazards each
      in `[0, 1]`

  - [ ]* 8.4 Write property test for cumulative-curve monotonicity and bounds
    - **Property 6: Cumulative-curve monotonicity and bounds**
    - **Validates: Requirements 6.3, 8.5**
    - Generate random `h in [0, 1]^13`; assert `F(a) = 1 - prod(1 - h(k))` is
      non-decreasing and within `[0, 1]`

  - [ ]* 8.5 Write property test for cohort timing dominance
    - **Property 7: Cohort timing dominance**
    - **Validates: Requirements 6.4**
    - Generate `h_B` and a non-negative increment to form `h_A >= h_B`; assert
      `F_A(a) >= F_B(a)` for all `a`

- [ ] 9. Checkpoint - Survival model core stable
  - Ensure all tests pass, ask the user if questions arise.

- [x] 10. Implement cohort aggregation and timing-uncertainty intervals (Enhancement 2)
  - [x] 10.1 Implement `aggregate_cohort_curves(F, cohort_week, approved)`
    - Average `F(a)` over approved applicants per cohort week (via
      `assign_cohort_week`); fall back to the mean curve over all approved
      applicants for empty cohorts; produce the 13x13 point trajectory
    - _Requirements: 6.3, 6.4, 8.2_

  - [x] 10.2 Implement `survival_intervals(...)`
    - For `N_BOOT_SURV` iterations, pick a bagged hazard model (timing-shape
      uncertainty) and resample the cohort's approved applicants with replacement
      (incidence uncertainty), recompute cohort curves, and take 5th/95th
      percentiles as `cdr_lower_90` / `cdr_upper_90`
    - _Requirements: 7.1, 7.2_

  - [ ]* 10.3 Write example test for feature-conditioned timing-shape differences
    - Build two materially different risk profiles; assert their timing shapes
      differ in at least one of the 13 intervals
    - _Requirements: 6.2_

- [ ] 11. Rework `build_deliverable_B` to use the survival model (Enhancement 2)
  - [x] 11.1 Replace `build_deliverable_B` body to orchestrate survival components
    - Call `expand_person_periods` -> `fit_hazard_models` ->
      `applicant_cumulative_curve` -> `aggregate_cohort_curves` ->
      `survival_intervals`; enforce per-cohort `np.maximum.accumulate` on point and
      bounds, `clamp_intervals(..., floor_width=0.0)`, integer `cohort_week` /
      `loan_age_weeks`, and write the 169-row CSV to
      `submission/submission_B_trajectory.csv`
    - Remove reliance on `estimate_timing_by_risk` for the new path (keep or retire
      the old helper as needed without affecting Deliverable A)
    - _Requirements: 6.3, 7.3, 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 9.6_

  - [ ]* 11.2 Write integration test for Deliverable B output contract
    - Assert `submission_B_trajectory.csv` exists with 169 rows covering the full
      13x13 grid, correct columns, integer cohort/age dtypes, values in `[0, 1]`,
      ordered bounds, and non-decreasing `cumulative_default_rate` within each cohort
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 7.3_

- [ ] 12. Wire everything into `main()` and verify the full submission (Enhancement 1 + 2)
  - [ ] 12.1 Confirm `main()` invokes reworked B and C with preserved A
    - Ensure `main()` still calls `build_deliverable_A` unchanged and routes
      artifacts into the reworked `build_deliverable_B` and `build_deliverable_C`;
      leave `submission_A_decisions.csv` name/location unchanged
    - _Requirements: 9.1, 9.2, 9.5, 9.6_

  - [ ]* 12.2 Write integration test for validator PASS and dependency/runtime limits
    - Run `python solution.py` then `python validate_submission.py ./submission`;
      assert `PASS` and exit 0; assert only numpy/pandas/sklearn imported by
      `solution.py`; assert the full run completes within ~2 minutes
    - _Requirements: 9.1, 9.3, 9.4_

  - [ ]* 12.3 Write integration test for Deliverable C output contract
    - Assert `submission_C_counterfactuals.csv` exists with one row per `query_id`,
      correct columns, values in `[0, 1]`, and `lower <= point <= upper` per row
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_

- [ ] 13. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional test sub-tasks and can be skipped for a faster
  MVP; core implementation tasks are never optional.
- Each task references specific requirement sub-clauses for traceability.
- Property tests use Hypothesis with a minimum of 100 examples and a tagging comment
  `# Feature: causal-and-survival-upgrades, Property {n}: {property_text}`.
- Each correctness property maps to exactly one property-based test (Properties 1-9).
- Checkpoints ensure incremental validation at natural break points.
- All source changes are confined to `solution.py`; tests live in `test_solution.py`.

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1", "2.1"] },
    { "id": 1, "tasks": ["2.2", "2.3", "3.1", "7.1"] },
    { "id": 2, "tasks": ["3.2", "4.1", "7.2", "8.1"] },
    { "id": 3, "tasks": ["4.2", "4.3", "4.4", "8.2"] },
    { "id": 4, "tasks": ["6.1", "8.3", "8.4", "8.5", "10.1"] },
    { "id": 5, "tasks": ["6.2", "6.3", "10.2", "10.3"] },
    { "id": 6, "tasks": ["11.1"] },
    { "id": 7, "tasks": ["11.2", "12.1"] },
    { "id": 8, "tasks": ["12.2", "12.3"] }
  ]
}
```
