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
