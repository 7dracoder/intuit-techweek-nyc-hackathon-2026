#!/usr/bin/env python3
"""Standalone submission format validator for the SMB Underwriting Challenge.

Run this on your submission folder BEFORE you upload. It checks that your four
files are named, structured, and formatted exactly as the automated scorer
expects. A submission that does not pass this gate cannot be scored and will be
disqualified -- so make sure it prints PASS.

    pip install -r requirements.txt
    python validate_submission.py path/to/your_submission_folder

Your submission folder must contain (flat, exact names):

    submission_A_decisions.csv
    submission_B_trajectory.csv
    submission_C_counterfactuals.csv
    submission_D_writeup.pdf        (the writeup is human-graded; a missing PDF
                                     is only a warning here, but you must include
                                     it in your real submission)

This script needs only numpy + pandas. It reads the expected applicant/query ID
sets from the ``expected_ids/`` folder shipped next to it -- you do NOT need the
full dataset downloaded for validation to work.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Expected submission schema (single source of truth)
# --------------------------------------------------------------------------- #

FILE_A = "submission_A_decisions.csv"
FILE_B = "submission_B_trajectory.csv"
FILE_C = "submission_C_counterfactuals.csv"
FILE_D = "submission_D_writeup.pdf"

COLUMNS_A = ("applicant_id", "decision", "predicted_pd", "pd_lower_90", "pd_upper_90")
COLUMNS_B = (
    "cohort_week",
    "loan_age_weeks",
    "cumulative_default_rate",
    "cdr_lower_90",
    "cdr_upper_90",
)
COLUMNS_C = ("query_id", "predicted_pd_cf", "pd_cf_lower_90", "pd_cf_upper_90")


class Level(str, Enum):
    ERROR = "error"
    WARN = "warn"


@dataclass(frozen=True)
class SubmissionIssue:
    deliverable: str
    code: str
    level: Level
    message: str


@dataclass
class SubmissionReport:
    issues: list[SubmissionIssue] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """True iff there are no ERROR-level issues (the scoring gate)."""
        return not any(i.level is Level.ERROR for i in self.issues)

    @property
    def n_error(self) -> int:
        return sum(1 for i in self.issues if i.level is Level.ERROR)

    @property
    def n_warn(self) -> int:
        return sum(1 for i in self.issues if i.level is Level.WARN)

    def add(self, deliverable: str, code: str, level: Level, message: str) -> None:
        self.issues.append(SubmissionIssue(deliverable, code, level, message))


# --------------------------------------------------------------------------- #
# Expected ID sets (from the shipped manifest, not the full dataset)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ExpectedSpec:
    applicant_ids: frozenset[str]
    query_ids: frozenset[str]
    n_cohort_weeks: int

    @classmethod
    def from_manifest(cls, expected_dir: Path) -> "ExpectedSpec":
        """Load expected IDs + grid size from the ``expected_ids/`` folder."""
        ed = Path(expected_dir)
        applicants = _read_id_file(ed / "applicant_ids.txt")
        queries = _read_id_file(ed / "query_ids.txt")
        manifest = json.loads((ed / "manifest.json").read_text())
        return cls(
            applicant_ids=frozenset(applicants),
            query_ids=frozenset(queries),
            n_cohort_weeks=int(manifest["n_cohort_weeks"]),
        )


def _read_id_file(path: Path) -> list[str]:
    return [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]


# --------------------------------------------------------------------------- #
# Low-level helpers
# --------------------------------------------------------------------------- #


def _read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def _numeric(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)


def _check_columns(
    report: SubmissionReport, deliverable: str, df: pd.DataFrame, expected: tuple[str, ...]
) -> bool:
    missing = [c for c in expected if c not in df.columns]
    if missing:
        report.add(
            deliverable,
            "missing_columns",
            Level.ERROR,
            f"missing required columns: {', '.join(missing)}",
        )
        return False
    # Unexpected columns are harmless to the scorer (it reads by name), but warn
    # so teams can catch typos / stray columns before submitting.
    extra = [c for c in df.columns if c not in expected]
    if extra:
        report.add(
            deliverable,
            "unexpected_columns",
            Level.WARN,
            f"unexpected extra column(s) will be ignored by scoring: {', '.join(extra)}",
        )
    return True


def _check_in_unit_interval(
    report: SubmissionReport, deliverable: str, df: pd.DataFrame, cols: tuple[str, ...]
) -> None:
    for col in cols:
        x = _numeric(df[col])
        if np.isnan(x).any():
            report.add(
                deliverable,
                "non_numeric",
                Level.ERROR,
                f"column '{col}' has non-numeric or missing values",
            )
            continue
        if (x < 0).any() or (x > 1).any():
            report.add(
                deliverable,
                "out_of_range",
                Level.ERROR,
                f"column '{col}' has values outside [0, 1]",
            )


def _check_interval_order(
    report: SubmissionReport,
    deliverable: str,
    df: pd.DataFrame,
    lower: str,
    point: str,
    upper: str,
) -> None:
    lo, mid, hi = _numeric(df[lower]), _numeric(df[point]), _numeric(df[upper])
    valid = ~(np.isnan(lo) | np.isnan(mid) | np.isnan(hi))
    bad = valid & ~((lo <= mid + 1e-9) & (mid <= hi + 1e-9))
    n_bad = int(bad.sum())
    if n_bad:
        report.add(
            deliverable,
            "interval_order",
            Level.ERROR,
            f"{n_bad} row(s) violate {lower} <= {point} <= {upper}",
        )


def _check_id_set(
    report: SubmissionReport,
    deliverable: str,
    submitted: pd.Series,
    expected: frozenset[str],
    id_name: str,
) -> None:
    sub = submitted.astype(str)
    if sub.duplicated().any():
        n_dup = int(sub.duplicated().sum())
        report.add(deliverable, "duplicate_ids", Level.ERROR, f"{n_dup} duplicate {id_name} value(s)")
    sub_set = frozenset(sub)
    missing = expected - sub_set
    extra = sub_set - expected
    if missing:
        report.add(
            deliverable,
            "missing_ids",
            Level.ERROR,
            f"{len(missing)} expected {id_name}(s) not present in submission",
        )
    if extra:
        report.add(
            deliverable,
            "unknown_ids",
            Level.ERROR,
            f"{len(extra)} {id_name}(s) not in the expected ID set",
        )


# --------------------------------------------------------------------------- #
# Per-deliverable validators
# --------------------------------------------------------------------------- #


def _validate_a(report: SubmissionReport, df: pd.DataFrame, spec: ExpectedSpec) -> None:
    if not _check_columns(report, "A", df, COLUMNS_A):
        return
    _check_id_set(report, "A", df["applicant_id"], spec.applicant_ids, "applicant_id")

    decision = _numeric(df["decision"])
    if np.isnan(decision).any():
        report.add("A", "non_numeric", Level.ERROR, "decision has non-numeric values")
    elif not np.isin(decision[~np.isnan(decision)], (0, 1)).all():
        report.add("A", "bad_decision", Level.ERROR, "decision must be 0 or 1")

    _check_in_unit_interval(report, "A", df, ("predicted_pd", "pd_lower_90", "pd_upper_90"))
    _check_interval_order(report, "A", df, "pd_lower_90", "predicted_pd", "pd_upper_90")


def _validate_b(report: SubmissionReport, df: pd.DataFrame, spec: ExpectedSpec) -> None:
    if not _check_columns(report, "B", df, COLUMNS_B):
        return

    n = spec.n_cohort_weeks
    expected_rows = n * n
    if len(df) != expected_rows:
        report.add(
            "B",
            "row_count",
            Level.ERROR,
            f"expected {expected_rows} rows ({n}x{n} grid), got {len(df)}",
        )

    cohort = _numeric(df["cohort_week"])
    age = _numeric(df["loan_age_weeks"])
    in_grid = (
        (cohort >= 1)
        & (cohort <= n)
        & (age >= 1)
        & (age <= n)
        & (cohort == np.floor(cohort))
        & (age == np.floor(age))
    )
    if not in_grid.all():
        report.add(
            "B",
            "bad_grid",
            Level.ERROR,
            f"cohort_week / loan_age_weeks must be integers in [1, {n}]",
        )
    else:
        pairs = pd.MultiIndex.from_arrays([cohort.astype(int), age.astype(int)])
        if pairs.duplicated().any():
            report.add("B", "duplicate_cells", Level.ERROR, "duplicate (cohort, age) cells")
        expected_pairs = {(w, a) for w in range(1, n + 1) for a in range(1, n + 1)}
        if set(map(tuple, pairs)) != expected_pairs:
            report.add("B", "incomplete_grid", Level.ERROR, "missing (cohort, age) cells")

    _check_in_unit_interval(
        report, "B", df, ("cumulative_default_rate", "cdr_lower_90", "cdr_upper_90")
    )
    _check_interval_order(
        report, "B", df, "cdr_lower_90", "cumulative_default_rate", "cdr_upper_90"
    )

    cdr = _numeric(df["cumulative_default_rate"])
    valid = ~(np.isnan(cohort) | np.isnan(age) | np.isnan(cdr))
    if valid.any():
        tmp = pd.DataFrame(
            {"w": cohort[valid].astype(int), "a": age[valid].astype(int), "cdr": cdr[valid]}
        ).sort_values(["w", "a"])
        bad_cohorts = 0
        for _, g in tmp.groupby("w"):
            if np.any(np.diff(g["cdr"].to_numpy()) < -1e-9):
                bad_cohorts += 1
        if bad_cohorts:
            report.add(
                "B",
                "non_monotone",
                Level.ERROR,
                f"{bad_cohorts} cohort(s) have a non-monotone (decreasing) trajectory",
            )


def _validate_c(report: SubmissionReport, df: pd.DataFrame, spec: ExpectedSpec) -> None:
    if not _check_columns(report, "C", df, COLUMNS_C):
        return
    _check_id_set(report, "C", df["query_id"], spec.query_ids, "query_id")
    _check_in_unit_interval(
        report, "C", df, ("predicted_pd_cf", "pd_cf_lower_90", "pd_cf_upper_90")
    )
    _check_interval_order(
        report, "C", df, "pd_cf_lower_90", "predicted_pd_cf", "pd_cf_upper_90"
    )


# --------------------------------------------------------------------------- #
# Top-level entry point
# --------------------------------------------------------------------------- #


def validate_submission(submission_dir: str | Path, spec: ExpectedSpec) -> SubmissionReport:
    """Validate a submission directory against the expected ID sets + schema."""
    sub = Path(submission_dir)
    report = SubmissionReport()

    frames: dict[str, pd.DataFrame] = {}
    for key, fname in {"A": FILE_A, "B": FILE_B, "C": FILE_C}.items():
        df = _read_csv(sub / fname)
        if df is None:
            report.add(
                "files",
                "missing_or_unparsable",
                Level.ERROR,
                f"required file '{fname}' is missing or not valid CSV",
            )
        else:
            frames[key] = df

    if not (sub / FILE_D).exists():
        report.add(
            "files",
            "missing_writeup",
            Level.WARN,
            f"writeup '{FILE_D}' not found (include it in your real submission)",
        )

    if "A" in frames:
        _validate_a(report, frames["A"], spec)
    if "B" in frames:
        _validate_b(report, frames["B"], spec)
    if "C" in frames:
        _validate_c(report, frames["C"], spec)

    return report


def _print_report(report: SubmissionReport) -> None:
    if report.issues:
        print("\nISSUES")
        print("-" * 78)
        for i in report.issues:
            tag = "ERROR" if i.level is Level.ERROR else "warn "
            print(f"  [{tag}] {i.deliverable:>5} / {i.code}: {i.message}")
        print("-" * 78)
    else:
        print("\nNo issues found.")

    if report.passed:
        print(f"\nRESULT: PASS  (0 errors, {report.n_warn} warning(s))")
        print("Your submission is correctly formatted and ready to upload.")
    else:
        print(f"\nRESULT: FAIL  ({report.n_error} error(s), {report.n_warn} warning(s))")
        print("Fix every ERROR above and re-run. A failing submission cannot be scored.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate an SMB Underwriting Challenge submission folder."
    )
    parser.add_argument("submission_dir", help="Folder containing your A/B/C CSVs (+ D PDF).")
    parser.add_argument(
        "--expected-ids",
        default=None,
        help="Path to the expected_ids/ folder (defaults to the one next to this script).",
    )
    args = parser.parse_args(argv)

    expected_dir = (
        Path(args.expected_ids)
        if args.expected_ids
        else Path(__file__).resolve().parent / "expected_ids"
    )
    if not (expected_dir / "manifest.json").exists():
        print(
            f"ERROR: could not find the expected_ids manifest at {expected_dir}.\n"
            "Run this script from inside the released/ folder, or pass --expected-ids.",
            file=sys.stderr,
        )
        return 2

    spec = ExpectedSpec.from_manifest(expected_dir)
    print(
        f"Expecting {len(spec.applicant_ids):,} applicants, "
        f"{len(spec.query_ids):,} queries, "
        f"{spec.n_cohort_weeks}x{spec.n_cohort_weeks} trajectory grid."
    )
    report = validate_submission(args.submission_dir, spec)
    _print_report(report)
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
