# Deliverable D - Technical Writeup

**Team:** <your team name>

## 1. Problem framing & assumptions violated

We are underwriting a fixed-terms SMB product (60-day term, daily ACH draws, 35%
APR, 3% origination fee). A loan defaults on 3 consecutive misses, 6 cumulative
misses, or a positive balance at day 90. Our job is a *profit* decision plus a
*timing* forecast plus *interventional* PDs - not a single classifier.

The dataset breaks several standard ML assumptions, and each changed our approach:

- **Selection bias / outcomes missing-not-at-random.** Outcomes exist *only* for
  loans a prior underwriter approved (51,722 of 85,340 train rows; the other
  33,618 declined rows have no label). We train on the approved-and-matured slice,
  which is not a random sample of applicants. We mitigate by keeping
  `prior_underwriter_score` and `prior_decision` as features (so the model can
  absorb part of the selection mechanism) and we treat the resulting PD as
  "PD on the population the prior policy would fund," flagged as a limitation.
- **Optimistically biased self-reported fields.** `stated_*` values are applicant
  claims; we keep them but lean on bank-feed and bureau signals where available.
- **Partial coverage (MNAR features).** Bank-feed columns are null for ~37% of
  rows (no linked feed). We do not impute a fake value; we let the
  gradient-boosted trees route NaN natively, and "has a feed" is itself signal.
- **Right-censoring / immature loans.** Timing is only meaningful for matured
  loans, so Deliverable B is framed as a survival/timing problem, not a yes/no.

## 2. Methodology

**PD model (A).** A bagged ensemble (8 members, bootstrap rows + distinct seeds)
of `HistGradientBoostingClassifier`, trained on all rows with an observed
`default_flag`. It handles NaN and integer-coded categoricals natively, so no
leakage-prone imputation is needed. We exclude all outcome columns and IDs;
`application_timestamp` is used only to derive the cohort. We address the
selection bias above with **reject inference via inverse-propensity weighting
(IPW)**: a propensity-of-approval model `e(X)=P(approved|X)` is fit on all train
rows, and labeled training rows are reweighted by `1/clip(e(X))` so the model
generalizes toward the full applicant population, not just the prior lender's
approved slice. We **keep IPW only if it does not degrade validation AUC and
holds calibration** (it improved AUC 0.7517 -> 0.7528). PD is **calibrated** with
isotonic regression on the labeled validation set (held out from training). Final
validation discrimination is AUC ~0.75 with near-perfect decile calibration.

**Decision rule (A).** We approve on **expected profit**, not a fixed PD cutoff.
With principal `L`: income if repaid `= L*(fee + APR*term/365)`; loss if default
`= L*(1 - recovery) - L*fee`. Crucially, recovery is **not** just
`final_recovered_amount` (~9% of principal) - a defaulter amortizes via daily
draws until it defaults (median day 37/60), so effective recovery
`= days_to_default/term + post-default recovery ~ 70%`. We fit a **per-loan
recovery (LGD) model** to predict this effective recovery from features, and we
choose the approval PD threshold by **simulating realized portfolio profit on
validation** (sweeping thresholds against true outcomes). Approve iff
`expected_profit > 0` AND `PD <= simulated_threshold (~0.22)`, giving a ~62%
approval rate that funds the profitable, low-PD book.

**Trajectory (B).** We separate *incidence* from *timing*.
`CDR(w, a) = incidence(w) * F_w(a)`, where `incidence(w)` is the mean calibrated
PD over **our approved** cohort-`w` applicants, and `F_w(a)` is a **risk-bucketed**
default-timing curve: we estimate the fraction of eventual defaults occurring by
week `a` separately for low/medium/high-risk buckets (forced non-decreasing), then
mix them per cohort by its risk composition, with a global-curve fallback for
thin buckets. This guarantees monotone trajectories (validator requirement) and
lets riskier cohorts default *faster*, not just *more*.

## 3. Causal reasoning & counterfactual methodology

The data is **observational**: features are correlated through unobserved
confounders and through engineered relationships, so a coefficient or a SHAP value
is an *associational* quantity, not a causal effect. An observational prediction
answers "what PD do applicants who *happen to have* feature = v show?"; an
**interventional** prediction answers `do(feature = v)` - "what PD if we *set*
this feature, holding the rest fixed and breaking the feature's incoming causal
arrows?" These differ whenever the feature shares a confounder with default.

Our counterfactual procedure (C) implements Pearl's three-step counterfactual on
a fitted **Structural Causal Model (SCM)**. We posit a small, domain-justified DAG
over the economic mediators (e.g. stated revenue -> observed monthly revenue ->
cash-balance floor -> overdrafts; debt & credit band -> utilization) and fit one
regression per child = f(parents). For `do(feature = v)` we: **(1) Abduct** -
compute each child's residual `u = actual - f(parents)` on the original row,
capturing the applicant's idiosyncratic noise; **(2) Act** - set the intervened
feature to `v`; **(3) Predict** - re-simulate every downstream child in
topological order as `f(new_parents) + u`, then re-score the calibrated ensemble.
The residual (abduction) step is essential: it guarantees a *no-op* intervention
(`v` = current value) leaves PD exactly unchanged, eliminating the
regression-to-the-mean offset that a naive "replace child with its prediction"
would inject. We verified this: no-op interventions move PD by 0.000, and real
interventions move it in causally correct directions (lowering delinquency or
utilization reduces PD; more overdrafts raises it), with ~20% up / ~21% down /
~58% near-zero across the 900 queries - the balanced signature of genuine
interventions rather than a biased re-prediction.

What we are giving up, stated honestly: our DAG covers the main economic
mediators but not every feature. Interventions on a feature with no modeled
children behave as a direct edit (its residual-preserving effect is exact for the
target but does not propagate). We do not claim to recover a unique true SCM from
observational data alone - the graph encodes assumptions we defend on domain
grounds, and we disclose them rather than hide them.

**Defending drivers to a regulator.** We would present monotone, calibrated,
documented drivers (utilization, overdrafts, cash-balance floor, delinquency,
prior defaults) with directions that match credit intuition, show the
observational-vs-interventional distinction explicitly, and disclose that
protected-class proxies are not used and that decisions are profit-threshold based
and reproducible - not a black-box cutoff.

## 4. Calibration & uncertainty quantification

All point PDs are isotonic-calibrated on held-out validation data. Intervals come
from the **ensemble spread**: per row we take the 5th/95th percentiles across the
8 bagged members, map them through the same isotonic calibrator, then apply a
**per-risk-bin (Mondrian-style) conformal widening** - a separate additive `delta`
per PD bin chosen on validation so that *binned* realized default rates fall inside
the band ~90% of the time. Correcting coverage per bin (rather than one global
delta) lets the intervals stay tight where the model is confident and widen only
where it is not. (With binary labels, per-row 0/1 coverage is not the right target;
we calibrate coverage of the *rate* within risk bins, which is what B is scored
on - and we observe 100% bin-wise coverage on validation at a mean width ~0.20.)
We clamp all bounds to [0,1], enforce `lower <= point <= upper`, and impose a small
floor width to avoid dishonest zero-width intervals. B intervals additionally come
from bootstrapping each cohort's incidence and inherit the monotone timing curve.
The tradeoff: tighter bands score better on width but risk under-coverage, so we
pick the smallest per-bin widening that reaches the coverage target.

## 5. Limitations & what we'd do differently

- **The SCM is partial and assumption-driven.** Our DAG covers the main economic
  mediators, not every feature, and the graph is justified on domain grounds rather
  than discovered from data. Interventions on features outside the graph act as
  direct edits. With more time we would run causal-discovery checks, expand the
  graph, and add sensitivity analysis to unobserved confounding.
- **Selection bias is only partly corrected.** IPW (reject inference) assumes
  selection-on-observables; if the prior lender used information we cannot see,
  residual bias remains. A two-stage (accept-then-default) or self-learning reject-
  inference scheme, and outcome validation on a truly random holdout, would go
  further. We also cannot directly measure calibration on the never-funded region
  because no outcomes exist there.
- **Timing curves use coarse risk buckets.** We mix three risk-bucket timing curves
  per cohort; a fully feature-conditioned discrete-time hazard (or Cox) model would
  give smoother, sharper per-cohort shapes and tighter B intervals.
- **Recovery model is point-estimate.** The per-loan LGD model feeds a single
  expected-recovery number; modeling the recovery *distribution* would propagate
  loss uncertainty into the decision and the intervals.
- **A single learner family.** We rely on gradient-boosted trees; stacking with a
  second model family and time-aware cross-validated tuning would likely add a few
  points of discrimination.

<!-- References (optional; do not count toward the 4-page body limit). -->
