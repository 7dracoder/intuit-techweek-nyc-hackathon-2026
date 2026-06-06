#!/usr/bin/env python3
"""
SMB Underwriting Challenge - full solution pipeline (upgraded).

Produces the three machine-scored deliverables into ./submission/ :

    submission_A_decisions.csv        (profit-optimal approve/decline + calibrated PD + 90% PI)
    submission_B_trajectory.csv       (13x13 cohort x loan-age cumulative default trajectory + 90% PI)
    submission_C_counterfactuals.csv  (do(feature=value) interventional PD via an SCM + 90% PI)

Upgrades over the baseline (see ACTION_AND_RESEARCH.md for the roadmap):
  * P1  Structural Causal Model (SCM) for Deliverable C - interventions propagate
        through fitted structural equations (abduction-free do-operator on a DAG)
        instead of editing a single deterministic mediator.
  * P2  Reject inference via Inverse-Propensity Weighting (IPW) for the PD model,
        run as a measured TOGGLE (kept only if it helps validation profit/calibration).
  * P2b Per-loan recovery (LGD) model feeding the expected-profit decision, plus a
        profit-simulated approval threshold.
  * P3  Risk-bucketed default-timing curves for Deliverable B (guarded fallback to
        the global curve when a bucket is data-thin).
  * P4  Per-risk-bin (Mondrian-style) conformal interval widening.

Engine: scikit-learn HistGradientBoosting (handles NaN + categoricals natively,
no OpenMP/GPU needed). Runs locally in ~1-2 minutes; also runs unchanged on Colab.

Run:
    pip install -r requirements.txt
    python solution.py
    python validate_submission.py ./submission
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, brier_score_loss

# --------------------------------------------------------------------------- #
# Paths, constants & toggles
# --------------------------------------------------------------------------- #

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "dataset"
OUT = ROOT / "submission"
OUT.mkdir(exist_ok=True)

RANDOM_SEED = 42
N_BAG = 8                      # ensemble members for uncertainty
TARGET_COVERAGE = 0.90         # nominal interval coverage
N_CONF_BINS = 10               # per-bin conformal bins

# Discrete-time hazard / survival model (Deliverable B) constants.
N_BAG_SURV = 8                 # bagged hazard models (timing-curve uncertainty)
N_BOOT_SURV = 200              # cohort applicant resamples (incidence uncertainty)
N_AGE_WEEKS = 13               # discrete loan-age intervals (weeks 1..13)
DTD_MIN, DTD_MAX = 1, 90       # days_to_default estimation window

USE_IPW = True                 # reject-inference toggle (auto-disabled if it hurts)

# Loan economics (from dataset/README.md "Loan product terms").
TERM_DAYS = 60
APR = 0.35
ORIG_FEE_RATE = 0.03
INTEREST_RATE_IF_PAID = APR * TERM_DAYS / 365.0  # ~0.0575

OUTCOME_COLS = [
    "default_flag", "days_to_default", "days_to_full_repayment",
    "repayment_status", "final_recovered_amount", "observation_status",
]
ID_COLS = ["business_id", "applicant_id"]
DROP_FROM_FEATURES = OUTCOME_COLS + ID_COLS + ["application_timestamp"]

# Leaky for the propensity model (null for every declined row).
PROPENSITY_DROP = ["prior_approved_amount"]


# --------------------------------------------------------------------------- #
# Data loading & feature prep
# --------------------------------------------------------------------------- #

def load_data():
    train = pd.read_csv(DATA / "train.csv")
    val = pd.read_csv(DATA / "validation.csv")
    test = pd.read_csv(DATA / "test.csv")
    data_dict = pd.read_csv(DATA / "data_dictionary.csv")
    cohorts = pd.read_csv(DATA / "cohort_week_definitions.csv")
    queries = pd.read_csv(DATA / "intervention_queries.csv")
    return train, val, test, data_dict, cohorts, queries


def get_feature_lists(data_dict: pd.DataFrame):
    feature_cols, categorical_cols = [], []
    for _, row in data_dict.iterrows():
        field = row["field"]
        if field in DROP_FROM_FEATURES:
            continue
        feature_cols.append(field)
        if str(row["dtype"]).strip().lower() == "categorical":
            categorical_cols.append(field)
    return feature_cols, categorical_cols


def prepare_features(df: pd.DataFrame, feature_cols, categorical_cols) -> pd.DataFrame:
    """Coerce all features to float (categoricals are integer-coded; NaN kept)."""
    X = df.reindex(columns=feature_cols).copy()
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce").astype("float64")
    return X


def assign_cohort_week(df: pd.DataFrame, cohorts: pd.DataFrame) -> pd.Series:
    ts = pd.to_datetime(df["application_timestamp"]).dt.normalize()
    cw = pd.Series(np.nan, index=df.index, dtype="float64")
    for _, r in cohorts.iterrows():
        start = pd.to_datetime(r["start_date"]); end = pd.to_datetime(r["end_date"])
        cw[(ts >= start) & (ts <= end)] = int(r["cohort_week"])
    return cw


# --------------------------------------------------------------------------- #
# Model training (bagged HistGradientBoosting; optional sample weights)
# --------------------------------------------------------------------------- #

CLF_PARAMS = dict(
    learning_rate=0.05, max_iter=500, max_leaf_nodes=48,
    min_samples_leaf=80, l2_regularization=2.0, max_bins=255,
    early_stopping=False,
)
REG_PARAMS = dict(
    learning_rate=0.05, max_iter=300, max_leaf_nodes=31,
    min_samples_leaf=50, l2_regularization=2.0, early_stopping=False,
)
# Lighter params for hazard models (large person-period dataset; stay in budget).
SURV_CLF_PARAMS = dict(
    learning_rate=0.05, max_iter=200, max_leaf_nodes=31,
    min_samples_leaf=100, l2_regularization=2.0, max_bins=128,
    early_stopping=False,
)


def train_bagged_models(X, y, categorical_cols, sample_weight=None):
    cat_mask = [c in categorical_cols for c in X.columns]
    models = []
    n = len(X)
    rng = np.random.RandomState(RANDOM_SEED)
    for b in range(N_BAG):
        idx = rng.randint(0, n, size=n)  # bootstrap rows
        clf = HistGradientBoostingClassifier(
            random_state=RANDOM_SEED + b, categorical_features=cat_mask, **CLF_PARAMS
        )
        sw = None if sample_weight is None else sample_weight[idx]
        clf.fit(X.iloc[idx], y[idx], sample_weight=sw)
        models.append(clf)
    return models


def ensemble_predict(models, X):
    preds = np.column_stack([m.predict_proba(X)[:, 1] for m in models])
    return preds.mean(axis=1), preds


# --------------------------------------------------------------------------- #
# P2 - Reject inference: inverse-propensity-of-approval weights
# --------------------------------------------------------------------------- #

def compute_ipw_weights(train, labeled_mask, feature_cols, categorical_cols):
    """
    Propensity e(X)=P(approved|X) from ALL train rows; inverse weights for the
    labeled (approved+matured) rows. Clipped for positivity, normalized to mean 1.
    """
    approved = (train["prior_decision"] == 1).astype(int).to_numpy()
    feats = [c for c in feature_cols if c not in PROPENSITY_DROP]
    Xs = prepare_features(train, feats, categorical_cols)
    cat_mask = [c in categorical_cols for c in Xs.columns]

    prop = HistGradientBoostingClassifier(
        random_state=RANDOM_SEED, categorical_features=cat_mask, **CLF_PARAMS
    )
    prop.fit(Xs, approved)
    e = prop.predict_proba(Xs)[:, 1]
    e_clip = np.clip(e, 0.05, 0.95)

    w = np.ones(len(train))
    w_lab = 1.0 / e_clip[labeled_mask]
    w_lab = w_lab / w_lab.mean()          # normalize -> stable effective sample size
    w[labeled_mask] = w_lab
    frac_low = float((e[approved == 0] < 0.05).mean())  # overlap diagnostic
    return w, e, frac_low


# --------------------------------------------------------------------------- #
# Calibration & per-bin conformal widening
# --------------------------------------------------------------------------- #

def fit_isotonic(raw_pd, y_true, sample_weight=None):
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(raw_pd, y_true, sample_weight=sample_weight)
    return iso


def tune_perbin_conformal(pd_point, pd_lo, pd_hi, y_true, target=TARGET_COVERAGE):
    """
    Return (edges, deltas): a per-PD-bin additive widening so realized default rate
    sits inside [lo-delta, hi+delta] for ~target of sub-bins within each PD bin.
    Mondrian-style: coverage is corrected across the PD range, not just on average.
    """
    edges = np.quantile(pd_point, np.linspace(0, 1, N_CONF_BINS + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    deltas = []
    for i in range(N_CONF_BINS):
        m = (pd_point >= edges[i]) & (pd_point < edges[i + 1])
        if m.sum() < 30:
            deltas.append(0.05)
            continue
        p = pd_point[m]; lo = pd_lo[m]; hi = pd_hi[m]; yt = y_true[m]
        order = np.argsort(p)
        p, lo, hi, yt = p[order], lo[order], hi[order], yt[order]
        k = max(2, min(8, m.sum() // 20))
        ed = np.linspace(0, len(p), k + 1).astype(int)
        br, bl, bh = [], [], []
        for j in range(k):
            a, b = ed[j], ed[j + 1]
            if b > a:
                br.append(yt[a:b].mean()); bl.append(lo[a:b].mean()); bh.append(hi[a:b].mean())
        br, bl, bh = np.array(br), np.array(bl), np.array(bh)
        chosen = 0.5
        for d in np.linspace(0.0, 0.5, 251):
            if ((br >= bl - d) & (br <= bh + d)).mean() >= target:
                chosen = float(d); break
        deltas.append(chosen)
    return edges, np.array(deltas)


def apply_perbin_conformal(pd_point, pd_lo, pd_hi, edges, deltas):
    lo = pd_lo.copy(); hi = pd_hi.copy()
    for i in range(len(deltas)):
        m = (pd_point >= edges[i]) & (pd_point < edges[i + 1])
        lo[m] -= deltas[i]; hi[m] += deltas[i]
    return lo, hi


def clamp_intervals(point, lo, hi, floor_width=0.02):
    point = np.clip(point, 0.0, 1.0); lo = np.clip(lo, 0.0, 1.0); hi = np.clip(hi, 0.0, 1.0)
    lo = np.minimum(lo, point); hi = np.maximum(hi, point)
    narrow = (hi - lo) < floor_width; half = floor_width / 2.0
    lo[narrow] = np.clip(point[narrow] - half, 0.0, 1.0)
    hi[narrow] = np.clip(point[narrow] + half, 0.0, 1.0)
    lo = np.minimum(lo, point); hi = np.maximum(hi, point)
    return point, lo, hi


# --------------------------------------------------------------------------- #
# P2b - Per-loan recovery (LGD) model
# --------------------------------------------------------------------------- #

def fit_recovery_model(train, feature_cols, categorical_cols):
    """
    Predict EFFECTIVE recovery rate on a (would-be) default:
        recovery = days_to_default/TERM_DAYS (pre-default daily draws)
                 + final_recovered_amount/requested_amount (post-default), clipped [0,1].
    Trained on observed defaults; used to refine per-loan loss in the profit rule.
    """
    dft = train[train["default_flag"] == 1].copy()
    pre = (dft["days_to_default"] / TERM_DAYS).clip(0, 1)
    post = (dft["final_recovered_amount"] / dft["requested_amount"]).clip(0, 1)
    rec = (pre + post).clip(0, 1).to_numpy()
    X = prepare_features(dft, feature_cols, categorical_cols)
    cat_mask = [c in categorical_cols for c in X.columns]
    reg = HistGradientBoostingRegressor(
        random_state=RANDOM_SEED, categorical_features=cat_mask, **REG_PARAMS
    )
    reg.fit(X, rec)
    return reg, float(rec.mean())


def expected_profit(pd_cal, principal, recovery_rate):
    profit_if_paid = principal * (ORIG_FEE_RATE + INTEREST_RATE_IF_PAID)
    loss_if_default = principal * (1.0 - recovery_rate) - principal * ORIG_FEE_RATE
    return (1.0 - pd_cal) * profit_if_paid - pd_cal * loss_if_default


# --------------------------------------------------------------------------- #
# P1 - Structural Causal Model (SCM) for Deliverable C
# --------------------------------------------------------------------------- #
# A pragmatic, domain-justified DAG over the economic mediators. Each (child:
# parents) entry is a structural equation child = f(parents) we fit from data.
# On do(X=v) we SET X (breaking its incoming edges) and re-simulate X's
# descendants in topological order, then score PD. Features not in the DAG fall
# back to the deterministic-ratio edit.
# Children are listed in a valid topological order (parents before children):
#   MR, DEBT, MULTI -> VOL, INQ -> TREND, PAY, UTIL, CASH -> OD
# 10 fitted children span three groups: 6 bank-feed, 3 bureau-credit, 1
# application-context. Every parent name is a `data_dictionary.csv` column.
SCM_EDGES = {
    # --- Bank-feed group (6 fitted children) ---
    "observed_monthly_revenue_avg_3mo": [
        "stated_annual_revenue", "sector", "employee_count_bucket",
        "vintage_years", "stated_time_in_business",
    ],
    # --- Bureau-credit group (3 fitted children) ---
    "existing_debt_obligations": [
        "stated_annual_revenue", "employee_count_bucket", "vintage_years",
        "owner_personal_credit_band",
    ],
    # --- Application-context group (1 fitted child) ---
    "multi_lender_inquiry_count_30d": [
        "application_channel", "days_since_last_inquiry_elsewhere",
        "owner_personal_credit_band",
    ],
    "observed_revenue_volatility": [
        "sector", "vintage_years", "employee_count_bucket",
        "observed_monthly_revenue_avg_3mo",
    ],
    "recent_inquiries_count_6mo": [
        "multi_lender_inquiry_count_30d", "owner_personal_credit_band",
        "days_since_last_external_decline",
    ],
    "observed_revenue_trend_3mo": [
        "observed_monthly_revenue_avg_3mo", "observed_revenue_volatility",
        "sector", "vintage_years",
    ],
    "payroll_regularity_score": [
        "observed_monthly_revenue_avg_3mo", "observed_revenue_volatility",
        "employee_count_bucket", "vintage_years",
    ],
    "aggregate_credit_utilization": [
        "existing_debt_obligations", "owner_personal_credit_band",
        "stated_annual_revenue", "recent_inquiries_count_6mo",
    ],
    "observed_cash_balance_p10": [
        "observed_monthly_revenue_avg_3mo", "observed_revenue_volatility",
        "existing_debt_obligations", "requested_amount",
    ],
    "observed_overdraft_count_3mo": [
        "observed_cash_balance_p10", "observed_revenue_volatility",
        "payroll_regularity_score",
    ],
}
# Deterministic engineered child. The stored ratio uses an unknown internal scale
# (it equals requested_amount / ANNUAL revenue, not monthly), so we never recompute
# it from a guessed formula - that would corrupt the value. Instead we scale the
# EXISTING stored ratio by the multiplicative change in its inputs under the
# intervention, which is scale-invariant and leaves a no-op intervention unchanged.
def _rescale_ratio(r_old, amt_old, rev_old, amt_new, rev_new):
    if pd.isna(r_old):
        return r_old
    factor = 1.0
    if pd.notna(amt_old) and pd.notna(amt_new) and amt_old not in (0, 0.0):
        factor *= amt_new / amt_old
    if pd.notna(rev_old) and pd.notna(rev_new) and rev_new not in (0, 0.0):
        factor *= rev_old / rev_new
    return r_old * factor


def _topo_order(edges):
    """Topological order of fitted children (parents before children)."""
    order, seen = [], set()
    def visit(node):
        if node in seen:
            return
        seen.add(node)
        for p in edges.get(node, []):
            if p in edges:
                visit(p)
        if node in edges:
            order.append(node)
    for n in edges:
        visit(n)
    return order


def fit_scm(train, categorical_cols):
    """Fit one regressor per structural-equation child."""
    scm = {}
    for child, parents in SCM_EDGES.items():
        rows = train[train[child].notna()].copy()
        if len(rows) < 200:
            continue
        Xp = prepare_features(rows, parents, categorical_cols)
        cat_mask = [c in categorical_cols for c in Xp.columns]
        reg = HistGradientBoostingRegressor(
            random_state=RANDOM_SEED, categorical_features=cat_mask, **REG_PARAMS
        )
        reg.fit(Xp, pd.to_numeric(rows[child], errors="coerce").astype(float))
        scm[child] = (parents, reg)
    return scm


def scm_intervene(row, feature, value, scm, categorical_cols):
    """
    do(feature=value) with abduction (Pearl's 3 steps):
      1. ABDUCT: for each fitted child, compute the applicant's residual
         u_child = actual_child - reg.predict(parents)  on the ORIGINAL row.
         This residual encodes the applicant's idiosyncratic noise term.
      2. ACT: set the intervened feature to `value`.
      3. PREDICT: re-simulate each downstream child in topo order as
         child = reg.predict(new_parents) + u_child, preserving the residual.

    The residual step is what makes a no-op intervention (value == current) leave
    PD exactly unchanged - it avoids the "regression-to-the-mean" offset you get if
    you naively replace a child with its predicted value. Deterministic children
    are recomputed exactly. Returns the modified row.
    """
    r = row.copy()

    # 1. ABDUCTION - residuals on the original (pre-intervention) row.
    residuals = {}
    for child, (parents, reg) in scm.items():
        actual = pd.to_numeric(pd.Series([row.get(child)]), errors="coerce").iloc[0]
        if pd.isna(actual):
            residuals[child] = None  # cannot abduct; leave child untouched
            continue
        Xp0 = prepare_features(pd.DataFrame([row]), parents, categorical_cols)
        residuals[child] = actual - float(reg.predict(Xp0)[0])

    # Capture inputs to the deterministic ratio BEFORE the intervention.
    amt_old = pd.to_numeric(pd.Series([row.get("requested_amount")]), errors="coerce").iloc[0]
    rev_old = pd.to_numeric(pd.Series([row.get("observed_monthly_revenue_avg_3mo")]), errors="coerce").iloc[0]
    ratio_old = pd.to_numeric(pd.Series([row.get("requested_amount_to_observed_revenue")]), errors="coerce").iloc[0]

    # 2. ACTION.
    r[feature] = value

    # 3. PREDICTION - re-simulate downstream children, adding back the residual.
    topo = _topo_order(SCM_EDGES)
    changed = {feature}
    for child in topo:
        if child == feature:
            changed.add(child)
            continue
        if residuals.get(child) is None:
            continue
        parents, reg = scm[child]
        if not (set(parents) & changed):
            continue
        Xp = prepare_features(pd.DataFrame([r]), parents, categorical_cols)
        r[child] = float(reg.predict(Xp)[0]) + residuals[child]
        changed.add(child)

    # Deterministic ratio: rescale the stored value by the input change (scale-safe).
    amt_new = pd.to_numeric(pd.Series([r.get("requested_amount")]), errors="coerce").iloc[0]
    rev_new = pd.to_numeric(pd.Series([r.get("observed_monthly_revenue_avg_3mo")]), errors="coerce").iloc[0]
    r["requested_amount_to_observed_revenue"] = _rescale_ratio(
        ratio_old, amt_old, rev_old, amt_new, rev_new
    )
    return r


# --------------------------------------------------------------------------- #
# Deliverable A
# --------------------------------------------------------------------------- #

def _eval(models, iso, X, y):
    raw, _ = ensemble_predict(models, X)
    p = iso.predict(raw)
    return roc_auc_score(y, p), brier_score_loss(y, p)


def build_deliverable_A(train, val, test, feature_cols, categorical_cols):
    print("\n[A] Training PD model ...")
    tr_lab_mask = train["default_flag"].notna().to_numpy()
    tr_lab = train[tr_lab_mask].copy()
    va_lab = val[val["default_flag"].notna()].copy()

    X_tr = prepare_features(tr_lab, feature_cols, categorical_cols)
    y_tr = tr_lab["default_flag"].astype(int).to_numpy()
    X_va = prepare_features(va_lab, feature_cols, categorical_cols)
    y_va = va_lab["default_flag"].astype(int).to_numpy()

    # Baseline (no IPW).
    base_models = train_bagged_models(X_tr, y_tr, categorical_cols)
    base_iso = fit_isotonic(ensemble_predict(base_models, X_va)[0], y_va)
    auc0, brier0 = _eval(base_models, base_iso, X_va, y_va)
    print(f"[A] no-IPW : AUC={auc0:.4f}  Brier={brier0:.4f}")

    models, iso = base_models, base_iso
    if USE_IPW:
        w_all, e, frac_low = compute_ipw_weights(train, tr_lab_mask, feature_cols, categorical_cols)
        w_tr = w_all[tr_lab_mask]
        ipw_models = train_bagged_models(X_tr, y_tr, categorical_cols, sample_weight=w_tr)
        ipw_iso = fit_isotonic(ensemble_predict(ipw_models, X_va)[0], y_va)
        auc1, brier1 = _eval(ipw_models, ipw_iso, X_va, y_va)
        print(f"[A] IPW    : AUC={auc1:.4f}  Brier={brier1:.4f}  "
              f"(declined-with-overlap-gap={frac_low:.3f})")
        # Keep IPW only if it does not degrade discrimination materially and helps/holds calibration.
        if (auc1 >= auc0 - 0.005) and (brier1 <= brier0 + 0.002):
            print("[A] -> keeping IPW (reject inference applied).")
            models, iso = ipw_models, ipw_iso
        else:
            print("[A] -> discarding IPW (did not help); using baseline.")

    # Per-loan recovery model (P2b).
    rec_model, rec_mean = fit_recovery_model(train, feature_cols, categorical_cols)
    print(f"[A] mean effective recovery = {rec_mean:.3f}")

    # Per-bin conformal widening tuned on validation.
    mean_va, mat_va = ensemble_predict(models, X_va)
    cal_va = iso.predict(mean_va)
    lo_va = iso.predict(np.quantile(mat_va, 0.05, axis=1))
    hi_va = iso.predict(np.quantile(mat_va, 0.95, axis=1))
    edges, deltas = tune_perbin_conformal(cal_va, lo_va, hi_va, y_va)
    print(f"[A] per-bin conformal deltas = {np.round(deltas, 3)}")

    # Profit-simulated threshold on validation (P2b). Decide on expected profit
    # using per-loan recovery; pick the PD cutoff maximizing realized val profit.
    rec_va = np.clip(rec_model.predict(X_va), 0, 1)
    prin_va = va_lab["requested_amount"].to_numpy()
    ep_va = expected_profit(cal_va, prin_va, rec_va)
    # realized profit if we approve everything with expected_profit>thr-shift; we
    # sweep a PD threshold and compute realized profit using TRUE outcomes.
    best_thr, best_profit = None, -np.inf
    for thr in np.linspace(0.05, 0.6, 56):
        appr = cal_va <= thr
        realized = np.where(
            y_va == 0,
            prin_va * (ORIG_FEE_RATE + INTEREST_RATE_IF_PAID),
            -(prin_va * (1.0 - rec_va) - prin_va * ORIG_FEE_RATE),
        )
        tot = realized[appr].sum()
        if tot > best_profit:
            best_profit, best_thr = tot, thr
    print(f"[A] profit-simulated PD threshold = {best_thr:.3f}")

    # Score the submission set = validation + test.
    sub = pd.concat([val, test], ignore_index=True)
    X_sub = prepare_features(sub, feature_cols, categorical_cols)
    mean_sub, mat_sub = ensemble_predict(models, X_sub)
    pd_cal = iso.predict(mean_sub)
    lo = iso.predict(np.quantile(mat_sub, 0.05, axis=1))
    hi = iso.predict(np.quantile(mat_sub, 0.95, axis=1))
    lo, hi = apply_perbin_conformal(pd_cal, lo, hi, edges, deltas)
    pd_cal, lo, hi = clamp_intervals(pd_cal, lo, hi)

    # Decision: profit-positive AND below the simulated threshold (both must hold).
    rec_sub = np.clip(rec_model.predict(X_sub), 0, 1)
    prin_sub = sub["requested_amount"].to_numpy()
    ep_sub = expected_profit(pd_cal, prin_sub, rec_sub)
    decision = ((ep_sub > 0) & (pd_cal <= best_thr)).astype(int)

    out = pd.DataFrame({
        "applicant_id": sub["applicant_id"].astype(str),
        "decision": decision.astype(int),
        "predicted_pd": pd_cal, "pd_lower_90": lo, "pd_upper_90": hi,
    })
    out.to_csv(OUT / "submission_A_decisions.csv", index=False)
    print(f"[A] approved {decision.mean()*100:.1f}% of {len(out)} -> submission_A_decisions.csv")

    return {
        "models": models, "iso": iso, "edges": edges, "deltas": deltas,
        "submission": sub, "X_sub": X_sub, "pd_cal": pd_cal, "decision": decision,
        "categorical_cols": categorical_cols, "feature_cols": feature_cols,
    }


# --------------------------------------------------------------------------- #
# Deliverable B  (P3: risk-bucketed timing)
# --------------------------------------------------------------------------- #

def estimate_timing_by_risk(train, val, models, iso, feature_cols, categorical_cols):
    """
    Estimate F_b(a) = fraction of defaults occurring by week a, within risk buckets.
    Risk is the calibrated PD on validation (out-of-training) defaulters; we use
    validation+train defaulter timing pooled per bucket, with a global fallback.
    Returns (global_F, bucket_edges, list_of_bucket_F).
    """
    dtd_all = train.loc[train["default_flag"] == 1, "days_to_default"].dropna().to_numpy()
    global_F = np.maximum.accumulate(np.array([(dtd_all <= 7 * a).mean() for a in range(1, 14)]))
    if global_F[-1] > 0:
        global_F = global_F / global_F[-1]

    # Score train defaulters with the (bagged) model for a risk proxy, then bucket.
    dft = train[train["default_flag"] == 1].copy()
    Xd = prepare_features(dft, feature_cols, categorical_cols)
    pd_d = iso.predict(ensemble_predict(models, Xd)[0])
    edges = np.quantile(pd_d, [0, 1 / 3, 2 / 3, 1.0]); edges[0], edges[-1] = -np.inf, np.inf
    bucket_F = []
    for i in range(3):
        m = (pd_d >= edges[i]) & (pd_d < edges[i + 1])
        d = dft.loc[m, "days_to_default"].dropna().to_numpy()
        if len(d) < 200:
            bucket_F.append(global_F); continue
        Fb = np.maximum.accumulate(np.array([(d <= 7 * a).mean() for a in range(1, 14)]))
        if Fb[-1] > 0:
            Fb = Fb / Fb[-1]
        bucket_F.append(Fb)
    return global_F, edges, bucket_F


def expand_person_periods(labeled, feature_cols):
    """
    Expand each labeled loan into discrete-time person-period rows over
    N_AGE_WEEKS (13) weekly intervals for discrete-time hazard modelling.

    Rules per row type:
    - **Defaulter** with event interval a* = min(13, ceil(days_to_default / 7)):
        emit rows for a = 1 .. a* where event = 0 for a < a* and event = 1 at a*.
        Defaulters with NaN days_to_default or days_to_default outside [DTD_MIN,
        DTD_MAX] are excluded entirely.
    - **Non-defaulter** (default_flag == 0, matured):
        emit N_AGE_WEEKS rows all with event = 0.

    Each row carries the loan's feature vector plus ``loan_age_weeks`` (= a).

    Returns
    -------
    X_pp : pd.DataFrame  – feature columns + ``loan_age_weeks`` (covariate)
    y_pp : np.ndarray    – binary event indicator (0/1) for each row
    age_pp : np.ndarray  – integer loan age a in {1..13} for each row
    """
    # Split into defaulters and non-defaulters.
    defaulters_raw = labeled[labeled["default_flag"] == 1].copy()
    non_defaulters = labeled[labeled["default_flag"] == 0].copy()

    # Filter defaulters: drop NaN days_to_default and those outside [DTD_MIN, DTD_MAX].
    dtd = pd.to_numeric(defaulters_raw["days_to_default"], errors="coerce")
    valid_mask = dtd.notna() & (dtd >= DTD_MIN) & (dtd <= DTD_MAX)
    defaulters = defaulters_raw[valid_mask].copy()
    dtd_valid = dtd[valid_mask].to_numpy()

    # --- Vectorised defaulter expansion -----------------------------------
    # a* = min(13, ceil(dtd / 7)) for each valid defaulter.
    a_star = np.minimum(N_AGE_WEEKS, np.ceil(dtd_valid / 7).astype(int))  # (n_def,)

    # Each defaulter i contributes a_star[i] rows (ages 1..a_star[i]).
    # Build a flat age vector and a flat loan-index vector using repeat.
    loan_idx_def = np.repeat(np.arange(len(defaulters)), a_star)  # row index into defaulters
    age_def = np.concatenate([np.arange(1, as_ + 1) for as_ in a_star]).astype(int)

    # event = 1 only at the last age for each defaulter.
    last_age_per_loan = np.repeat(a_star, a_star)
    event_def = (age_def == last_age_per_loan).astype(int)

    X_def = defaulters[feature_cols].iloc[loan_idx_def].reset_index(drop=True)
    X_def["loan_age_weeks"] = age_def

    # --- Vectorised non-defaulter expansion --------------------------------
    n_nd = len(non_defaulters)
    # Each non-defaulter contributes exactly N_AGE_WEEKS rows, all event=0.
    loan_idx_nd = np.repeat(np.arange(n_nd), N_AGE_WEEKS)
    age_nd = np.tile(np.arange(1, N_AGE_WEEKS + 1), n_nd).astype(int)
    event_nd = np.zeros(n_nd * N_AGE_WEEKS, dtype=int)

    X_nd = non_defaulters[feature_cols].iloc[loan_idx_nd].reset_index(drop=True)
    X_nd["loan_age_weeks"] = age_nd

    # --- Concatenate --------------------------------------------------------
    X_pp = pd.concat([X_def, X_nd], ignore_index=True)
    y_pp = np.concatenate([event_def, event_nd])
    age_pp = np.concatenate([age_def, age_nd])

    return X_pp, y_pp, age_pp


def fit_hazard_models(X_pp, y_pp, categorical_cols, n_bag=N_BAG_SURV):
    """
    Fit ``n_bag`` bagged HistGradientBoostingClassifiers on bootstrapped
    person-period rows for discrete-time hazard estimation.

    The ``loan_age_weeks`` covariate is treated as **numeric** (it is not in
    ``categorical_cols``). All other features reuse the existing categorical mask
    derived from ``categorical_cols``.

    Each fitted classifier predicts h(a | x) = P(event=1 | features, loan_age=a),
    i.e. the conditional hazard in [0, 1] for the given loan-age interval.

    When the person-period dataset is very large (> 200 000 rows), a stratified
    subsample is used per bag (all event rows + up to 5x that many non-event rows)
    to keep fitting within the runtime budget while preserving the class ratio.

    Parameters
    ----------
    X_pp : pd.DataFrame
        Person-period feature matrix including the ``loan_age_weeks`` column.
    y_pp : array-like of int (0/1)
        Binary event indicator for each person-period row.
    categorical_cols : list[str]
        Names of categorical features (``loan_age_weeks`` is intentionally absent
        so that age is treated as numeric).
    n_bag : int
        Number of bootstrap bags (default: N_BAG_SURV).

    Returns
    -------
    list[HistGradientBoostingClassifier]
        Length-``n_bag`` list of fitted classifiers.
    """
    # Build categorical mask aligned to X_pp columns.
    # loan_age_weeks is numeric, so it is excluded from the mask.
    cat_mask = [col in categorical_cols for col in X_pp.columns]

    y_arr = np.asarray(y_pp, dtype=int)
    n = len(X_pp)
    rng = np.random.RandomState(RANDOM_SEED)

    # Stratified subsampling indices: keep all events, subsample non-events to
    # cap the per-bag training set at ~100 000 rows (for runtime budget).
    MAX_BAG_ROWS = 100_000
    event_idx = np.where(y_arr == 1)[0]
    nonevent_idx = np.where(y_arr == 0)[0]
    n_events = len(event_idx)
    n_nonevent_cap = min(len(nonevent_idx), MAX_BAG_ROWS - n_events)
    use_subsample = (n > MAX_BAG_ROWS)

    models = []
    for b in range(n_bag):
        if use_subsample:
            # Stratified subsample per bag: all events + subsampled non-events.
            ne_sample = rng.choice(nonevent_idx, size=n_nonevent_cap, replace=False)
            base_idx = np.concatenate([event_idx, ne_sample])
            # Bootstrap within the subsample.
            idx = rng.choice(base_idx, size=len(base_idx), replace=True)
        else:
            # Small dataset: standard row-level bootstrap.
            idx = rng.randint(0, n, size=n)
        clf = HistGradientBoostingClassifier(
            random_state=RANDOM_SEED + b,
            categorical_features=cat_mask,
            **SURV_CLF_PARAMS,
        )
        clf.fit(X_pp.iloc[idx], y_arr[idx])
        models.append(clf)

    return models


def applicant_cumulative_curve(model, X_app, categorical_cols):
    """
    Compute the cumulative default fraction F(a) for each applicant across
    all N_AGE_WEEKS = 13 discrete loan-age intervals using a single fitted
    hazard classifier.

    For each applicant, 13 covariate rows are built by tiling the applicant's
    feature vector and appending ``loan_age_weeks = 1, 2, ..., 13``.  The
    classifier's ``predict_proba`` yields the conditional hazard
    ``h(a | features)`` for each interval.  Hazards are clipped to [0, 1]
    before computing the survival product:

        S(a) = prod_{k=1}^{a} (1 - h(k))
        F(a) = 1 - S(a)

    Because each ``h(k)`` is in [0, 1], ``S(a)`` is non-increasing in ``a``,
    so ``F(a)`` is non-decreasing and lies in [0, 1] by construction
    (Requirements 6.1, 6.3).

    Parameters
    ----------
    model : HistGradientBoostingClassifier
        A single fitted hazard classifier (one member of the bag returned by
        ``fit_hazard_models``).
    X_app : pd.DataFrame, shape (n_applicants, n_features)
        Applicant feature rows **without** a ``loan_age_weeks`` column.
    categorical_cols : list[str]
        Names of categorical features.  ``loan_age_weeks`` is intentionally
        absent so that the age covariate is treated as numeric.

    Returns
    -------
    F : np.ndarray, shape (n_applicants, N_AGE_WEEKS)
        Cumulative default fraction F(a) for each applicant at each loan age.
        Each row is non-decreasing and lies in [0, 1].
    """
    ages = np.arange(1, N_AGE_WEEKS + 1)          # [1, 2, ..., 13]
    n_app = len(X_app)

    # Build the (n_applicants * 13) x (n_features + 1) prediction frame.
    # Tile each applicant row 13 times, then append the age covariate.
    tiled = pd.DataFrame(
        np.repeat(X_app.values, N_AGE_WEEKS, axis=0),
        columns=X_app.columns,
    )
    tiled["loan_age_weeks"] = np.tile(ages, n_app)

    # Predict P(event=1 | features, loan_age_weeks) — take the positive class.
    hazards_flat = model.predict_proba(tiled)[:, 1]          # shape (n_app * 13,)
    hazards_flat = np.clip(hazards_flat, 0.0, 1.0)

    # Reshape to (n_applicants, 13): row i, col j => h(j+1 | applicant i).
    H = hazards_flat.reshape(n_app, N_AGE_WEEKS)             # shape (n_app, 13)

    # Survival: S(a) = prod_{k=0}^{a-1} (1 - H[:, k]), computed cumulatively.
    survival = np.cumprod(1.0 - H, axis=1)                   # shape (n_app, 13)

    # Cumulative default fraction F(a) = 1 - S(a).
    F = 1.0 - survival                                       # shape (n_app, 13)

    return F


def aggregate_cohort_curves(F, cohort_week, approved):
    """
    Average cumulative default curves F(a) over approved applicants per cohort
    week to produce the 13x13 point trajectory.

    For each cohort week ``w in {1..13}``, the curve is the mean of ``F[i]``
    over all approved applicants ``i`` assigned to week ``w``.  If no approved
    applicants belong to cohort week ``w`` (empty cohort), the function falls
    back to the mean curve over *all* approved applicants (Requirements 6.3,
    6.4, 8.2).

    Parameters
    ----------
    F : np.ndarray, shape (n_applicants, N_AGE_WEEKS)
        Cumulative default curves for **all** submission applicants (i.e. the
        full set passed to the B pipeline), output of
        ``applicant_cumulative_curve``.
    cohort_week : np.ndarray, shape (n_applicants,)
        Integer cohort-week assignments (1..13) for every applicant in ``F``.
    approved : np.ndarray of bool, shape (n_applicants,)
        Boolean mask indicating which applicants were approved.  Only approved
        applicants contribute to cohort averages.

    Returns
    -------
    cohort_curves : np.ndarray, shape (13, N_AGE_WEEKS)
        ``cohort_curves[w-1]`` is the mean ``F(a)`` trajectory for cohort week
        ``w``.  All rows are non-decreasing (inherited from F) and lie in
        [0, 1].
    """
    F = np.asarray(F)                   # (n_applicants, 13)
    approved = np.asarray(approved, dtype=bool)
    cohort_week = np.asarray(cohort_week, dtype=int)

    # Pre-compute fallback: mean over *all* approved applicants.
    if approved.sum() > 0:
        fallback = F[approved].mean(axis=0)   # shape (13,)
    else:
        fallback = np.zeros(N_AGE_WEEKS)

    cohort_curves = np.empty((13, N_AGE_WEEKS), dtype=float)

    for w in range(1, 14):
        mask = approved & (cohort_week == w)
        if mask.sum() > 0:
            cohort_curves[w - 1] = F[mask].mean(axis=0)
        else:
            cohort_curves[w - 1] = fallback

    return cohort_curves


def survival_intervals(hazard_models, F_all, cohort_week, approved,
                       X_app, categorical_cols):
    """
    Bootstrap-based 90 % timing-uncertainty intervals for each (cohort_week,
    loan_age_weeks) grid cell.

    Two sources of uncertainty are propagated simultaneously in each of the
    ``N_BOOT_SURV = 200`` iterations (Requirements 7.1, 7.2):

      (a) **Timing-shape uncertainty**: a bagged hazard model is picked
          uniformly at random from ``hazard_models`` to recompute F for the
          resampled applicants.
      (b) **Incidence uncertainty**: approved applicants within each cohort
          week are resampled with replacement before averaging to form the
          cohort curve.

    The 5th and 95th percentiles over the 200 trajectories become
    ``cdr_lower_90`` and ``cdr_upper_90`` respectively.

    Parameters
    ----------
    hazard_models : list[HistGradientBoostingClassifier]
        Bagged hazard classifiers returned by ``fit_hazard_models``.
    F_all : np.ndarray, shape (n_applicants, N_AGE_WEEKS)
        The *point-estimate* cumulative curves (not used for interval
        computation; kept for interface consistency; may be None).
    cohort_week : np.ndarray, shape (n_applicants,)
        Integer cohort-week assignments for every submission applicant.
    approved : np.ndarray of bool, shape (n_applicants,)
        Boolean mask over submission applicants.
    X_app : pd.DataFrame, shape (n_applicants, n_features)
        Applicant feature rows **without** ``loan_age_weeks``.
    categorical_cols : list[str]
        Categorical feature names (passed through to
        ``applicant_cumulative_curve``).

    Returns
    -------
    lo : np.ndarray, shape (13, N_AGE_WEEKS)
        5th-percentile cohort trajectory (lower 90% bound).
    hi : np.ndarray, shape (13, N_AGE_WEEKS)
        95th-percentile cohort trajectory (upper 90% bound).
    """
    rng = np.random.RandomState(RANDOM_SEED)
    approved = np.asarray(approved, dtype=bool)
    cohort_week = np.asarray(cohort_week, dtype=int)

    # Pre-compute F for ALL applicants for each hazard model so that the
    # bootstrap loop only resamples precomputed arrays (no re-predictions).
    # Shape: (n_models, n_applicants, N_AGE_WEEKS).
    n_models = len(hazard_models)
    F_per_model = np.stack(
        [applicant_cumulative_curve(m, X_app, categorical_cols) for m in hazard_models],
        axis=0,
    )  # (n_models, n_applicants, 13)

    # Pre-compute per-cohort approved indices (for efficient per-cohort resampling).
    cohort_indices = {}
    for w in range(1, 14):
        idx = np.where(approved & (cohort_week == w))[0]
        cohort_indices[w] = idx

    all_approved_idx = np.where(approved)[0]   # fallback for empty cohorts

    # Collect bootstrap trajectories: shape (N_BOOT_SURV, 13, N_AGE_WEEKS).
    boot_curves = np.empty((N_BOOT_SURV, 13, N_AGE_WEEKS), dtype=float)

    for b in range(N_BOOT_SURV):
        # (a) Pick a random hazard model (timing-shape uncertainty).
        m_idx = rng.randint(0, n_models)
        F_m = F_per_model[m_idx]  # (n_applicants, 13)

        # (b) For each cohort week, resample approved applicants with replacement
        #     and average the precomputed F values (no re-prediction needed).
        for w in range(1, 14):
            idx = cohort_indices[w]
            if len(idx) == 0:
                idx = all_approved_idx          # fallback: all approved applicants

            # Resample with replacement (incidence uncertainty).
            res_idx = rng.choice(idx, size=len(idx), replace=True)
            boot_curves[b, w - 1] = F_m[res_idx].mean(axis=0)

    # 5th / 95th percentile over the bootstrap dimension.
    lo = np.percentile(boot_curves, 5, axis=0)   # shape (13, N_AGE_WEEKS)
    hi = np.percentile(boot_curves, 95, axis=0)  # shape (13, N_AGE_WEEKS)

    return lo, hi


def build_deliverable_B(artifacts, train, val, cohorts):
    print("\n[B] Building cohort default trajectories (discrete-time hazard survival model) ...")
    feature_cols = artifacts["feature_cols"]
    categorical_cols = artifacts["categorical_cols"]
    decision = artifacts["decision"]
    sub = artifacts["submission"]

    # 1. Extract labeled training loans (default_flag not null).
    labeled = train[train["default_flag"].notna()].copy()

    # 2. Expand labeled loans into person-period rows.
    print("[B] Expanding person-period rows ...")
    X_pp, y_pp, age_pp = expand_person_periods(labeled, feature_cols)

    # 3. Build pp_feature_cols = feature_cols + ["loan_age_weeks"].
    pp_feature_cols = list(feature_cols) + ["loan_age_weeks"]

    # 4. Fit bagged hazard models on person-period rows.
    print(f"[B] Fitting {N_BAG_SURV} hazard models on {len(X_pp)} person-period rows ...")
    hazard_models = fit_hazard_models(X_pp, y_pp, categorical_cols, n_bag=N_BAG_SURV)

    # 5. Get approved applicants from artifacts.
    approved_mask = (decision == 1)

    # Prepare applicant features for the submission set (without loan_age_weeks).
    X_app = prepare_features(sub, feature_cols, categorical_cols)

    # 6. Compute point-estimate F_all by averaging cumulative curves across all
    #    N_BAG_SURV hazard models.
    print("[B] Computing applicant cumulative curves (point estimate) ...")
    F_sum = np.zeros((len(X_app), N_AGE_WEEKS), dtype=float)
    for model in hazard_models:
        F_sum += applicant_cumulative_curve(model, X_app, categorical_cols)
    F_all = F_sum / N_BAG_SURV                       # shape (n_applicants, 13)

    # 7. Compute cohort_week assignments for all submission applicants.
    cohort_week = assign_cohort_week(sub, cohorts).to_numpy()

    # 8. Aggregate per-cohort point-estimate curves.
    print("[B] Aggregating cohort curves ...")
    cohort_curves = aggregate_cohort_curves(F_all, cohort_week, approved_mask)  # (13, 13)

    # 9. Compute timing-uncertainty intervals via bootstrap.
    print(f"[B] Computing survival intervals ({N_BOOT_SURV} bootstrap iterations) ...")
    lo, hi = survival_intervals(
        hazard_models, F_all, cohort_week, approved_mask, X_app, categorical_cols
    )  # both shape (13, 13)

    # 10. Enforce per-cohort monotonicity (np.maximum.accumulate) on point, lo, hi.
    for w_idx in range(13):
        cohort_curves[w_idx] = np.maximum.accumulate(cohort_curves[w_idx])
        lo[w_idx] = np.maximum.accumulate(lo[w_idx])
        hi[w_idx] = np.maximum.accumulate(hi[w_idx])

    # 11. Clamp intervals to guarantee ordered in-range bounds.
    # Flatten to 1-D, clamp, then reshape back.
    pt_flat = cohort_curves.ravel()
    lo_flat = lo.ravel()
    hi_flat = hi.ravel()
    pt_flat, lo_flat, hi_flat = clamp_intervals(pt_flat, lo_flat, hi_flat, floor_width=0.0)
    cohort_curves = pt_flat.reshape(13, N_AGE_WEEKS)
    lo = lo_flat.reshape(13, N_AGE_WEEKS)
    hi = hi_flat.reshape(13, N_AGE_WEEKS)

    # 12. Write the 169-row CSV.
    rows = []
    for w_idx in range(13):
        w = w_idx + 1
        for a_idx in range(N_AGE_WEEKS):
            a = a_idx + 1
            rows.append((
                w, a,
                float(cohort_curves[w_idx, a_idx]),
                float(lo[w_idx, a_idx]),
                float(hi[w_idx, a_idx]),
            ))

    b = pd.DataFrame(rows, columns=[
        "cohort_week", "loan_age_weeks",
        "cumulative_default_rate", "cdr_lower_90", "cdr_upper_90",
    ])
    b["cohort_week"] = b["cohort_week"].astype(int)
    b["loan_age_weeks"] = b["loan_age_weeks"].astype(int)
    b = b.sort_values(["cohort_week", "loan_age_weeks"]).reset_index(drop=True)
    b.to_csv(OUT / "submission_B_trajectory.csv", index=False)
    print(f"[B] wrote {len(b)}-row grid -> submission_B_trajectory.csv")
    return b


# --------------------------------------------------------------------------- #
# Deliverable C  (P1: SCM-based interventions)
# --------------------------------------------------------------------------- #

def report_directional_effects(results, sub):
    """Print-only diagnostic: per-query tally of upward / downward / near-zero
    counterfactual PD movements relative to each applicant's baseline PD.

    Parameters
    ----------
    results : list of 5-tuples (query_id, predicted_pd_cf, pd_cf_lower_90,
              pd_cf_upper_90, baseline_pd)
        The raw counterfactual rows collected inside build_deliverable_C (before
        conformal widening).  Each tuple includes the pre-intervention baseline
        PD as its 5th element.
    sub : pd.DataFrame indexed by applicant_id
        Unused — baseline_pd is embedded directly in each result tuple.
        Kept in the signature for documentation consistency.

    Prints an aggregate summary: share of queries that moved PD upward, downward,
    or near-zero (|delta| <= 1e-6).  Nothing is written to disk.
    """
    NEAR_ZERO = 1e-6
    n_up = n_down = n_flat = 0
    for item in results:
        _qid, pd_cf, _lo, _hi, baseline_pd = item
        delta = pd_cf - baseline_pd
        if abs(delta) <= NEAR_ZERO:
            n_flat += 1
        elif delta > 0:
            n_up += 1
        else:
            n_down += 1
    total = n_up + n_down + n_flat
    if total == 0:
        print("[C] report_directional_effects: no queries to report.")
        return
    print(
        f"[C] Directional effects over {total} queries: "
        f"upward={n_up} ({n_up/total:.1%}), "
        f"downward={n_down} ({n_down/total:.1%}), "
        f"near-zero={n_flat} ({n_flat/total:.1%})"
    )


def build_deliverable_C(artifacts, train, queries, feature_cols, categorical_cols):
    print("\n[C] Scoring counterfactuals via SCM ...")
    models, iso = artifacts["models"], artifacts["iso"]
    edges, deltas = artifacts["edges"], artifacts["deltas"]
    sub = artifacts["submission"].set_index("applicant_id")

    scm = fit_scm(train, categorical_cols)
    print(f"[C] fitted {len(scm)} structural equations: {list(scm.keys())}")

    results = []  # 5-tuples: (qid, pd_cf, lo, hi, baseline_pd)
    for _, q in queries.iterrows():
        qid, aid = q["query_id"], q["applicant_id"]
        feat, val = q["feature_name"], q["intervention_value"]
        if aid not in sub.index:
            results.append((qid, 0.5, 0.4, 0.6, 0.5)); continue
        row = sub.loc[aid]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        baseline_pd = float(row["predicted_pd"]) if "predicted_pd" in row.index else 0.5
        row = row.copy()

        row_cf = scm_intervene(row, feat, val, scm, categorical_cols)
        X_row = prepare_features(pd.DataFrame([row_cf]), feature_cols, categorical_cols)
        mean_cf, mat_cf = ensemble_predict(models, X_row)
        pd_cf = float(iso.predict(mean_cf)[0])
        lo = float(iso.predict(np.quantile(mat_cf, 0.05, axis=1))[0])
        hi = float(iso.predict(np.quantile(mat_cf, 0.95, axis=1))[0])
        results.append((qid, pd_cf, lo, hi, baseline_pd))

    # Print directional-effects diagnostic (Req 4.3) — no written output changed.
    report_directional_effects(results, sub)

    # Strip the 5th element (baseline_pd) before building the output DataFrame.
    rows_4 = [(qid, pd_cf, lo, hi) for qid, pd_cf, lo, hi, _ in results]
    c = pd.DataFrame(rows_4, columns=["query_id", "predicted_pd_cf", "pd_cf_lower_90", "pd_cf_upper_90"])
    lo2, hi2 = apply_perbin_conformal(
        c["predicted_pd_cf"].to_numpy(), c["pd_cf_lower_90"].to_numpy(),
        c["pd_cf_upper_90"].to_numpy(), edges, deltas)
    pt, lo2, hi2 = clamp_intervals(c["predicted_pd_cf"].to_numpy(), lo2, hi2)
    c["predicted_pd_cf"], c["pd_cf_lower_90"], c["pd_cf_upper_90"] = pt, lo2, hi2
    c.to_csv(OUT / "submission_C_counterfactuals.csv", index=False)
    print(f"[C] wrote {len(c)} rows -> submission_C_counterfactuals.csv")
    return c


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    print("Loading data ...")
    train, val, test, data_dict, cohorts, queries = load_data()
    feature_cols, categorical_cols = get_feature_lists(data_dict)
    print(f"{len(feature_cols)} features ({len(categorical_cols)} categorical).")

    artifacts = build_deliverable_A(train, val, test, feature_cols, categorical_cols)
    build_deliverable_B(artifacts, train, val, cohorts)
    build_deliverable_C(artifacts, train, queries, feature_cols, categorical_cols)

    print("\nDone. Validate with:")
    print(f"    python validate_submission.py {OUT}")


if __name__ == "__main__":
    main()
