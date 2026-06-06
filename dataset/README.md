# Dataset Guide

This folder contains the data for the **SMB Underwriting Challenge**. You are a
small-business lender: each row is a loan application, and your job is to decide
whom to fund, forecast how those loans repay over time, and answer causal
what-if questions about them.


## Getting the data

The three main tables are zipped to keep the download small. Unzip first:

```bash
unzip dataset-compressed.zip      # -> train.csv, validation.csv, test.csv
```

## Files

| File | Rows x Cols | What it is |
|---|---|---|
| `train.csv` | 85,340 x 44 | Historical applications. Repayment outcomes are filled in **only** for loans the prior lender approved and that have since matured; declined/immature loans have blank outcome fields. |
| `validation.csv` | 4,489 x 44 | Applications **with** outcomes filled in. Use this to tune and calibrate. |
| `test.csv` | 8,817 x 44 | The applications you are scored on. Features only -- **all outcome fields are blank** (withheld). |
| `data_dictionary.csv` | one row per field | Every column: name, type, group, whether it is an intervenable quantity, and a short description. |
| `intervention_queries.csv` | 900 | The causal what-if queries for Deliverable C: `(query_id, applicant_id, feature_name, intervention_value)`. |
| `cohort_week_definitions.csv` | 13 | Maps each `cohort_week` (1-13) to its exact calendar date range. Use this to assign test applications to cohorts for Deliverable B. |
| `submission_B_template.csv` | 169 | The pre-built 13 x 13 grid for Deliverable B. Overwrite the prediction columns; do not change the grid. |

You decide on the **validation + test** applicants combined (13,306 in total) for
Deliverable A.

## Columns

See `data_dictionary.csv` for the full list. Columns fall into these groups:

- **business_identity** -- who the business is (sector, geography, size, age, ids).
- **self_reported** -- values the applicant states on the application.
- **bank_feed** -- signals from a linked bank feed. These are blank when the
  applicant did not link a feed (`has_linked_bank_feed = False`).
- **bureau_credit** -- credit-bureau signals (utilization, inquiries, debt, band).
- **platform_engagement** -- how the business uses the lending platform.
- **application_context** -- when and how the application arrived, and prior history.
- **prior_underwriter** -- the previous lender's score and decision.
- **outcome** -- repayment results. Present only for approved + matured loans;
  always blank in `test.csv`.

Some fields are legitimately blank (null) -- e.g. bank-feed columns when no feed
is linked, or "days since last decline" when there was none. The data dictionary
notes which fields can be null.

## Loan product terms

Every loan, if funded, uses the same fixed terms:

- Amount: the `requested_amount`.
- Term: 60 days, repaid via daily ACH draws.
- APR: 35% (annualized).
- Origination fee: 3% of the amount, collected up front.

**Default definition.** A funded loan is repaid by **daily ACH draws** over the
60-day term (each day's draw either succeeds or is *missed* in full -- there are
no partial payments). A loan is counted as **defaulted** if **any** of the
following happens:

1. **3 consecutive missed draws** -- a successful draw resets this consecutive
   counter (a borrower can "cure"); or
2. **6 missed draws in total** over the life of the loan -- this cumulative
   counter never resets (it catches chronic, every-few-days unreliability); or
3. the **outstanding balance is still greater than zero at day 90** (the
   default window) -- i.e. the loan was never paid off in time.

`days_to_default` (when present) is the day in [1, 90] on which the loan first
met one of these conditions. A loan that is fully repaid before any of them is
`paid_in_full`. You will need this definition to predict default for
Deliverable A and the default *timing* for Deliverable B.

For what you must submit and how it is scored, see
[`../PARTICIPANT_INSTRUCTIONS.md`](../PARTICIPANT_INSTRUCTIONS.md).
