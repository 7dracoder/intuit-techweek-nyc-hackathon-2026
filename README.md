# SMB Underwriting Challenge - Participant Instructions

> ## STEP 0 (REQUIRED, DO THIS FIRST): Register your team
>
> **Register here as early as possible: [Form Link](https://forms.gle/9VPr8e7KNkZyDBUR6)
>
> You will **not** receive a submission link until you register. Don't leave this
> to the last minute.
>
> Provide your **team name** and, for each member (1-4 members per team):
> full name (required), email (required). After you
> register, we email your team a private link where you upload your submission.

---

## The challenge

You are a small-business lender. Using a historical book of loan applications,
build a model that (A) decides whom to fund, (B) forecasts how those loans
default over time, and (C) answers causal "what-if" questions - then defend your
reasoning in a short writeup (D).

## What you get

Everything is in this folder. Start with the dataset guide:
[`dataset/README.md`](dataset/README.md).

| File | Purpose |
|---|---|
| `dataset/dataset-compressed.zip` | The data: `train.csv`, `validation.csv`, `test.csv` (unzip first). |
| `dataset/data_dictionary.csv` | Every column: name, type, description. |
| `dataset/intervention_queries.csv` | The ~900 queries for Deliverable C. |
| `dataset/cohort_week_definitions.csv` | `cohort_week` (1-13) -> calendar date ranges (for Deliverable B). |
| `dataset/submission_B_template.csv` | The 169-row grid to fill for Deliverable B. |
| `submission_D_writeup_template.md` | Skeleton for the Deliverable D writeup. |
| `validate_submission.py` | Run this on your submission before uploading. |
| `requirements.txt` | The (minimal) Python deps to read data + validate. |

## The four deliverables

You submit **exactly four files**, with **exactly these names**. Scoring is
automated by joining on these names and IDs, so a wrong name or a missing ID
means your submission cannot be scored.

### A - `submission_A_decisions.csv` (your lending policy)

Your approve/decline decision plus your calibrated probability of default (PD)
for every applicant.

- **Rows:** one per applicant in `validation.csv` + `test.csv` (13,306 total).
- **Columns:**
  | column | meaning |
  |---|---|
  | `applicant_id` | matches the dataset |
  | `decision` | `1` = approve at the requested amount, `0` = decline |
  | `predicted_pd` | your PD point estimate in `[0, 1]` - **required for everyone, including declines** |
  | `pd_lower_90` | lower bound of your 90% interval on `predicted_pd` |
  | `pd_upper_90` | upper bound of your 90% interval |
- **Rule:** `pd_lower_90 <= predicted_pd <= pd_upper_90` on every row.

### B - `submission_B_trajectory.csv` (the default-timing forecast)

This is **not** a single default number - it is the *shape* of how defaults
accumulate over the life of a loan, per origination cohort. It is what tells us
whether you modeled loan timing rather than just a yes/no classifier.

For each origination **cohort week** `w` (1-13) and **loan age** `a` weeks (1-13),
predict the cumulative fraction of **your approved cohort-`w` loans** that have
defaulted **by day `7a`**.

- **Rows:** the full 13 x 13 = **169** grid. Use `dataset/submission_B_template.csv`:
  overwrite the three prediction columns; do not change the grid itself.
- **Columns:**
  | column | meaning |
  |---|---|
  | `cohort_week` | origination cohort 1-13 (from `cohort_week_definitions.csv`) |
  | `loan_age_weeks` | loan age 1-13 weeks after origination |
  | `cumulative_default_rate` | predicted cumulative default fraction by day `7a`, in `[0, 1]` |
  | `cdr_lower_90` / `cdr_upper_90` | your 90% interval bounds |
- **Rules:** `cdr_lower_90 <= cumulative_default_rate <= cdr_upper_90`, and within
  each cohort the `cumulative_default_rate` must be **non-decreasing as age
  increases** (cumulative rates can only go up).
- **Example row:** `cohort_week=3, loan_age_weeks=4, cumulative_default_rate=0.05`
  means "5% of my approved week-3 loans defaulted within the first 28 days."

### C - `submission_C_counterfactuals.csv` (the causal what-if)

`dataset/intervention_queries.csv` lists ~900 queries, each
`(query_id, applicant_id, feature_name, intervention_value)`. For each query,
predict the applicant's PD **if that one feature were *set* to that value by
intervention**, holding everything else fixed - i.e. `do(feature = value)`.

- **Rows:** one per `query_id` in `intervention_queries.csv`.
- **Columns:**
  | column | meaning |
  |---|---|
  | `query_id` | matches the queries file |
  | `predicted_pd_cf` | your post-intervention PD in `[0, 1]` |
  | `pd_cf_lower_90` / `pd_cf_upper_90` | your 90% interval bounds |
- **Rule:** `pd_cf_lower_90 <= predicted_pd_cf <= pd_cf_upper_90`.

### D - `submission_D_writeup.pdf` (the technical writeup)

A short writeup defending your methodology. Human-reviewed; reviewers grade on
substance, not polish.

- **Start from** [`submission_D_writeup_template.md`](submission_D_writeup_template.md),
  fill in each section, then **export to PDF** named `submission_D_writeup.pdf`.
- **Format (enforced):** max **4 pages** of body (excluding references), **>=11pt**
  font, **>=0.75 inch** margins. Content past page 4 is truncated for review.
- **Required sections, in this order:**
  1. Problem framing & assumptions violated
  2. Methodology
  3. Causal reasoning & counterfactual methodology
  4. Calibration & uncertainty quantification
  5. Limitations & what we'd do differently
- Section 3 (causal reasoning - distinguishing observational from interventional
  prediction, and how you'd defend your drivers to a regulator) carries the most
  weight.

## How you're scored (high level)

The A/B/C files are scored automatically; the writeup is reviewed by humans:

- **Portfolio profitability** from your approve/decline decisions (A).
- **Cohort-timing accuracy** of your trajectory forecast (B).
- **Calibration** of your 90% intervals (A and B).
- **Counterfactual accuracy** of your what-if predictions (C).
- **Technical writeup** (D).

We deliberately do not publish the exact scoring weights or formulas.

## Submission checklist

1. **Register** on the Google Form (Step 0) early - your private upload link is
   emailed to your team.
2. **Unzip and explore** the dataset (`dataset/`).
3. **Build your four files** with the exact names above. Use
   `submission_B_template.csv` for B and `submission_D_writeup_template.md` for D.
4. **Put all four files flat in one folder** (no subfolders).
5. **Validate** until it prints `PASS`:
   ```bash
   pip install -r requirements.txt
   python validate_submission.py path/to/your_submission_folder
   ```
6. **Upload** the four files to your team's private link from the email. Done.

## Hard requirements (read this)

- **Exact file names**, exactly as listed above. A typo means your file is not
  found and not scored.
- **Flat folder** - the four files directly in one folder, no nesting.
- **`validate_submission.py` must print `PASS`.** Format, naming, ID-coverage,
  range, and monotonicity mismatches are caught here. Because scoring for some deliverables are objective and fully
  automated, a submission that does not pass the validator will be disqualified.
