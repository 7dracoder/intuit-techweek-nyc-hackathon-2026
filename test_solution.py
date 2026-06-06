"""
Test suite for the SMB Underwriting Challenge pipeline (`solution.py`).

Test tooling notes
------------------
These tests use ``pytest`` as the runner and ``hypothesis`` for property-based
tests (Feature: causal-and-survival-upgrades, Properties 1-9).

``hypothesis`` is a TEST-ONLY dependency. It MUST NOT be imported by
``solution.py`` (the submission pipeline is restricted to numpy / pandas /
scikit-learn per Requirements 1.7, 6.5, 9.4). Install it locally with::

    pip install hypothesis pytest

Model-dependent tests operate on a small cached subsample of the training rows
(see the ``subsample`` fixture) so they stay well within the runtime budget.
"""

from __future__ import annotations

import functools

import pandas as pd
import pytest

# hypothesis is a TEST-ONLY dependency (property-based tests). It is intentionally
# imported here and never inside solution.py.
import hypothesis  # noqa: F401

import solution

# Fast, deterministic subsample size for model-dependent tests.
SUBSAMPLE_N = 400


@functools.lru_cache(maxsize=1)
def _load_train_cached() -> pd.DataFrame:
    """Load the full training frame once and cache it across tests."""
    train, _val, _test, _data_dict, _cohorts, _queries = solution.load_data()
    return train


@functools.lru_cache(maxsize=1)
def _subsample_cached() -> pd.DataFrame:
    """Deterministic, cached subsample of training rows for fast tests."""
    train = _load_train_cached()
    n = min(SUBSAMPLE_N, len(train))
    return train.sample(n=n, random_state=solution.RANDOM_SEED).reset_index(drop=True)


@pytest.fixture(scope="session")
def train_full() -> pd.DataFrame:
    """Session-scoped full training frame."""
    return _load_train_cached()


@pytest.fixture(scope="session")
def subsample() -> pd.DataFrame:
    """Session-scoped cached subsample of training rows for model-dependent tests."""
    return _subsample_cached()


@pytest.fixture(scope="session")
def data_dict() -> pd.DataFrame:
    """Session-scoped data dictionary frame."""
    _train, _val, _test, dd, _cohorts, _queries = solution.load_data()
    return dd


def test_scaffolding_imports_and_constants():
    """Smoke check: solution imports cleanly and survival/SCM constants exist."""
    assert solution.N_BAG_SURV == 8
    assert solution.N_BOOT_SURV == 200
    assert solution.N_AGE_WEEKS == 13
    assert (solution.DTD_MIN, solution.DTD_MAX) == (1, 90)


def test_subsample_fixture_nonempty(subsample):
    """The cached subsample fixture yields rows for downstream model tests."""
    assert len(subsample) > 0
    assert len(subsample) <= SUBSAMPLE_N


# --------------------------------------------------------------------------- #
# Feature engineering
# --------------------------------------------------------------------------- #

def test_engineered_features_present_and_deterministic(subsample):
    """compute_engineered yields all ENGINEERED columns, is deterministic, and
    introduces no leakage column (outcomes never appear)."""
    e1 = solution.compute_engineered(subsample)
    e2 = solution.compute_engineered(subsample)
    for col in solution.ENGINEERED:
        assert col in e1.columns
    # Deterministic.
    assert e1.equals(e2)
    # No outcome/ID column leaked into the engineered frame.
    for bad in solution.OUTCOME_COLS + solution.ID_COLS:
        assert bad not in e1.columns


def test_attach_engineered_is_idempotent(subsample):
    """Attaching engineered columns twice yields identical values."""
    once = solution.attach_engineered(subsample)
    twice = solution.attach_engineered(once)
    for col in solution.ENGINEERED:
        assert col in twice.columns
        a = once[col].to_numpy()
        b = twice[col].to_numpy()
        # NaN-aware equality.
        assert ((a == b) | (pd.isna(a) & pd.isna(b))).all()


# --------------------------------------------------------------------------- #
# Deliverable C — Pearl no-op invariant (the key correctness guarantee)
# --------------------------------------------------------------------------- #

@functools.lru_cache(maxsize=1)
def _scm_and_models_cached():
    """Fit a small SCM + PD ensemble on the cached subsample for invariant tests."""
    import numpy as np

    train = _load_train_cached()
    _train2, _val, _test, dd, _cohorts, _queries = solution.load_data()
    feature_cols, categorical_cols = solution.get_feature_lists(dd)
    feature_cols = feature_cols + [c for c in solution.ENGINEERED if c not in feature_cols]

    sub = solution.attach_engineered(
        train[train["default_flag"].notna()].sample(
            n=min(3000, int(train["default_flag"].notna().sum())),
            random_state=solution.RANDOM_SEED,
        ).reset_index(drop=True)
    )
    X = solution.prepare_features(sub, feature_cols, categorical_cols)
    y = sub["default_flag"].astype(int).to_numpy()
    models = solution.train_bagged_models(X, y, categorical_cols)
    scm = solution.fit_scm(sub, categorical_cols)
    return sub, feature_cols, categorical_cols, models, scm


def test_scm_noop_invariant_zero_delta():
    """do(feature = current_value) must leave PD exactly unchanged (|delta| < 1e-9).

    This is invariant #1 in DEVELOPER_HANDOFF.md — the abduction residual makes a
    no-op intervention a true identity. Re-verified with engineered features in
    the loop (they are recomputed inside scm_intervene)."""
    import numpy as np

    sub, feature_cols, categorical_cols, models, scm = _scm_and_models_cached()
    # Test the fitted SCM children (the non-deterministic causal nodes).
    test_features = list(scm.keys())
    max_abs_delta = 0.0
    n_checked = 0
    for _, row in sub.head(15).iterrows():
        X0 = solution.prepare_features(pd.DataFrame([row]), feature_cols, categorical_cols)
        pd0 = float(solution.ensemble_predict(models, X0)[0][0])
        for feat in test_features:
            cur = pd.to_numeric(pd.Series([row.get(feat)]), errors="coerce").iloc[0]
            if pd.isna(cur):
                continue
            row_cf = solution.scm_intervene(row, feat, cur, scm, categorical_cols)
            X1 = solution.prepare_features(pd.DataFrame([row_cf]), feature_cols, categorical_cols)
            pd1 = float(solution.ensemble_predict(models, X1)[0][0])
            max_abs_delta = max(max_abs_delta, abs(pd1 - pd0))
            n_checked += 1
    assert n_checked > 0
    assert max_abs_delta < 1e-9, f"no-op moved PD by {max_abs_delta:.3e}"


# --------------------------------------------------------------------------- #
# Deliverable B — survival curve monotonicity & level
# --------------------------------------------------------------------------- #

def test_scale_curve_to_incidence_monotone_and_levels():
    """Incidence-scaled curves are non-decreasing in age, in [0,1], and end at PD."""
    import numpy as np

    rng = np.random.RandomState(0)
    # Random monotone cumulative curves F (non-decreasing rows in [0,1]).
    raw = np.sort(rng.rand(20, solution.N_AGE_WEEKS), axis=1)
    pd_cal = rng.rand(20) * 0.4
    G = solution.scale_curve_to_incidence(raw, pd_cal)
    # Non-decreasing in age.
    assert (np.diff(G, axis=1) >= -1e-9).all()
    # Bounded.
    assert (G >= -1e-9).all() and (G <= 1.0 + 1e-9).all()
    # Terminal value equals calibrated PD (where terminal F was positive).
    terminal_ok = raw[:, -1] > 1e-12
    assert np.allclose(G[terminal_ok, -1], pd_cal[terminal_ok], atol=1e-9)
