"""Fit weights on the decision log, and prove they beat the ones set by feel.

Everything before this point encoded taste as constants a human chose. This
replaces them with numbers fitted to actual approve/reject decisions — which is
the whole reason the log exists.

Three disciplines, because a fit is easy and a *trustworthy* fit is not:

**Cross-validate, always.** A logistic regression with four predictors and forty
positives can fit the training data well and predict nothing. Every number
reported here comes from folds the model did not see.

**Compare against the incumbent, not against chance.** "Better than a coin
flip" is a low bar the current hand-set weights already clear. The question is
whether the fit beats *them*, scored the same way on the same held-out clips.
On the first real log it did not, and not narrowly: within rides the hand-set
weights scored 0.767 against the fit's 0.541. A linear model cannot represent
the power mean the composite uses, and "outstanding at one feature" is exactly
what that nonlinearity exists to express.

**Judge on the within-ride axis.** Pooled AUC is dominated by which ride a clip
came from — a 48% approval rate against 12% between two groups of rides — and
the composite was never meant to capture that. Pooled numbers made the fit look
like an improvement and the incumbent look useless; both were artefacts.

**Report the spread across folds.** Coefficients from a small sample move
around. A weight that flips sign between folds has not been measured, it has
been guessed with extra steps, and it should be reported as such rather than
averaged into false confidence.

No new dependency: scipy is already here for the roughness filter, and a
regularised logistic regression is twenty lines of `minimize`.
"""
from __future__ import annotations

import json
from typing import Any

import numpy as np
from scipy.optimize import minimize

# Two feature sets, because dropping rows with a missing feature is not a
# neutral act. `speed` is NaN on every ride shot without GPS, and an earlier
# version of this file silently discarded those rows — 52 of 193 decisions and
# 26 of 43 approvals, because those older bikejoring rides are exactly the ones
# whose clips get approved. The model was then fitted on the part of the library
# its owner likes least, and reported "no signal" from a biased sample.
#
# So: fit on what every row has, and fit the wider set separately on the subset
# that has it. Never quietly on whichever rows survive.
COMMON = ("turn", "rough", "jump")          # available on every ride
FEATURES = ("speed", "turn", "rough", "jump")
L2 = 1.0                 # ridge penalty; small samples need the shrinkage
FOLDS = 5


def load(conn, feats: tuple[str, ...] = FEATURES
         ) -> tuple[np.ndarray, np.ndarray, list[dict], dict]:
    """Feature matrix, approved flags, rows, and what had to be dropped.

    The drop counts are returned rather than logged and forgotten: a fit on a
    biased subsample looks exactly like a fit on all the data.
    """
    rows, X, y = [], [], []
    dropped, dropped_pos = 0, 0
    # Join the ride in: the grouping that matters for confounding is the ride,
    # not the file. A ride spans several chapters and they share whatever makes
    # that ride's clips appealing.
    for s in conn.execute(
            """SELECT seg.*, a.ride_id AS ride_id FROM segment seg
               JOIN asset a ON a.content_hash = seg.content_hash
               WHERE seg.status IN ('approved','rejected')"""):
        try:
            vec = json.loads(s["features"] or "{}")
        except json.JSONDecodeError:
            continue
        # A missing jump value means no airtime data, which is a real zero.
        # A missing speed value means no GPS, which is not — drop those rather
        # than teach the model that "no GPS" looks like "stationary".
        v = [vec.get(f) for f in feats]
        v = [0.0 if (f == "jump" and (x is None or not np.isfinite(x))) else x
             for f, x in zip(feats, v)]
        if any(x is None or not np.isfinite(x) for x in v):
            dropped += 1
            dropped_pos += (s["status"] == "approved")
            continue
        X.append(v)
        y.append(1.0 if s["status"] == "approved" else 0.0)
        rows.append(dict(s))
    return (np.array(X, dtype=float), np.array(y, dtype=float), rows,
            {"dropped": dropped, "dropped_approved": dropped_pos})


def _fit(X: np.ndarray, y: np.ndarray, l2: float = L2) -> np.ndarray:
    """Ridge-regularised logistic regression. Returns [intercept, *coefs]."""
    def loss(w):
        z = np.clip(w[0] + X @ w[1:], -30, 30)
        ll = np.sum(y * z - np.logaddexp(0, z))
        return -ll + l2 * np.sum(w[1:] ** 2)

    res = minimize(loss, np.zeros(X.shape[1] + 1), method="L-BFGS-B")
    return res.x


def _auc(scores: np.ndarray, y: np.ndarray) -> float:
    """Probability a random approved clip outranks a random rejected one.

    Rank-based, so it does not care about calibration — which matters because
    the two things being compared here are on completely different scales.
    """
    pos, neg = scores[y == 1], scores[y == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order), dtype=float)
    ranks[order] = np.arange(1, len(order) + 1)
    # average ranks for ties, or a flat scorer looks better than it is
    allv = np.concatenate([pos, neg])
    for v in np.unique(allv):
        m = allv == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    return (ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def _auc_within(scores: np.ndarray, y: np.ndarray,
                groups: np.ndarray) -> tuple[float, int]:
    """AUC counted only over pairs from the same ride.

    This is the number that decides whether a feature is real. Rides differ
    enormously in how often their clips get approved — 48% against 12% here —
    so any feature that merely runs higher on the rides you like will score well
    pooled while predicting nothing about *which clip within a ride* you take.
    A scorer knowing only which group a clip came from reached AUC 0.71 on this
    log, beating the fitted model's 0.66; pooled AUC cannot tell those apart and
    this can.
    """
    num = den = 0.0
    for g in np.unique(groups):
        m = groups == g
        pos, neg = scores[m & (y == 1)], scores[m & (y == 0)]
        if not len(pos) or not len(neg):
            continue
        comp = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
        num += comp
        den += len(pos) * len(neg)
    # The pair count is returned, not just the ratio. Only rides holding both an
    # approved and a rejected clip contribute, and at a 12% approval rate most
    # rides hold neither — so this number can be tiny while the AUC still prints
    # to three decimals and looks authoritative.
    return (float(num / den) if den else float("nan")), int(den)


def _folds(y: np.ndarray, k: int, seed: int = 0) -> list[np.ndarray]:
    """Stratified folds — with a 22% approval rate, random folds can land with
    almost no positives in them and the AUC becomes undefined."""
    rng = np.random.default_rng(seed)
    out = [[] for _ in range(k)]
    for label in (0.0, 1.0):
        idx = np.flatnonzero(y == label)
        rng.shuffle(idx)
        for i, j in enumerate(idx):
            out[i % k].append(j)
    return [np.array(sorted(f)) for f in out]


def incumbent_scores(rows: list[dict]) -> np.ndarray:
    """What the current weights + sharpness say about each clip.

    Uses the stored `score`, which is the composite the selector actually
    produced — the fairest possible statement of the incumbent's opinion.
    """
    return np.array([r["score"] if r["score"] is not None else np.nan
                     for r in rows], dtype=float)


def group_rates(conn) -> list[dict]:
    """Approval rate per ride, with whether that ride had GPS.

    Here because the decision log's strongest pattern turned out not to be a
    feature at all: which ride a clip came from predicted approval with an odds
    ratio of 7.3 (p = 2e-08), against 3.1 for the best feature contrast. A
    per-clip model cannot see that, and reports "no signal" while a very large
    signal sits one level up.
    """
    rows = conn.execute(
        """SELECT a.ride_id AS ride, a.lighting AS lighting, s.features AS feats,
                  s.status AS status
           FROM segment s JOIN asset a ON a.content_hash = s.content_hash
           WHERE s.status IN ('approved','rejected')""").fetchall()
    agg: dict[Any, dict] = {}
    for r in rows:
        # JSON has no NaN, so a missing speed round-trips as null -> None, and
        # np.isfinite(None) raises rather than returning False.
        try:
            sp = json.loads(r["feats"] or "{}").get("speed")
        except json.JSONDecodeError:
            sp = None
        has_speed = isinstance(sp, (int, float)) and np.isfinite(sp)
        g = agg.setdefault(r["ride"], {"ride": r["ride"], "lighting": r["lighting"],
                                       "gps": has_speed, "n": 0, "ok": 0})
        g["n"] += 1
        g["ok"] += (r["status"] == "approved")
        g["gps"] = g["gps"] or has_speed
    return sorted(agg.values(), key=lambda g: -(g["ok"] / max(g["n"], 1)))


# Ride-level attributes worth testing against approval. All are recorded at
# ingest already, so this costs one query and no reprocessing.
ATTRIBUTES = (
    ("lighting", "lighting"),
    ("aspect", "aspect"),
    ("camera", "camera_model"),
    ("horizon locked", "horizon_locked"),
    ("month", "substr(recorded_at, 1, 7)"),
    ("has GPS", "CASE WHEN gps_lat IS NULL THEN 'no' ELSE 'yes' END"),
    # Included so the circularity is visible: `lighting` comes from sun
    # elevation when GPS gives a position and from exposure response when it
    # does not, so on a library split by GPS era it cannot disagree with the
    # split. An attribute computed differently per group explains nothing.
    ("lighting from", "lighting_source"),
)


def _partition(conn, expr: str) -> frozenset:
    """Which rides fall in which bucket, as a comparable signature."""
    rows = conn.execute(
        f"""SELECT DISTINCT a.ride_id AS ride, {expr} AS k FROM segment s
            JOIN asset a ON a.content_hash = s.content_hash
            WHERE s.status IN ('approved','rejected')""").fetchall()
    buckets: dict[Any, set] = {}
    for r in rows:
        buckets.setdefault(r["k"], set()).add(r["ride"])
    return frozenset(frozenset(v) for v in buckets.values())


def confounded_groups(conn) -> list[list[str]]:
    """Attributes that cut the library in exactly the same place.

    Two attributes inducing an identical partition of rides are one variable
    with two names, and presenting them as separate evidence triples the
    apparent support for a single observation. On this library GPS era, aspect
    ratio, month and lighting all coincide — nothing was ever shot 8:7 in
    spring — so no analysis of the log can separate them. That is a fact about
    the data collection, not something a better model fixes.
    """
    sigs: dict[frozenset, list[str]] = {}
    for label, expr in ATTRIBUTES:
        try:
            part = _partition(conn, expr)
        except Exception:
            continue
        if len(part) > 1:
            sigs.setdefault(part, []).append(label)
    return [v for v in sigs.values() if len(v) > 1]


def attribute_rates(conn) -> dict[str, list[dict]]:
    """Approval rate broken down by every ride attribute already on record.

    The decision log's biggest effect was a 7.3x gap between rides with and
    without GPS — but GPS is an era marker, and eras carry everything else with
    them: aspect ratio, time of day, season, which trails were being ridden.
    Picking one of those by eye is how a correlate gets mistaken for a cause, so
    this reports all of them side by side and lets the counts arbitrate.
    """
    out: dict[str, list[dict]] = {}
    for label, expr in ATTRIBUTES:
        try:
            rows = conn.execute(
                f"""SELECT {expr} AS k,
                           SUM(s.status = 'approved') AS ok,
                           COUNT(*) AS n,
                           COUNT(DISTINCT a.ride_id) AS rides
                    FROM segment s JOIN asset a ON a.content_hash = s.content_hash
                    WHERE s.status IN ('approved','rejected')
                    GROUP BY k ORDER BY 1.0 * ok / COUNT(*) DESC""").fetchall()
        except Exception:
            continue
        vals = [dict(r) for r in rows if r["n"]]
        # A breakdown with one bucket explains nothing, and one where every
        # bucket is a single ride is just the ride effect wearing a new label.
        if len(vals) > 1 and sum(v["n"] for v in vals) > 0:
            out[label] = vals
    return out


def evaluate(conn, folds: int = FOLDS, seed: int = 0,
             feats: tuple[str, ...] = FEATURES) -> dict[str, Any]:
    X, y, rows, drops = load(conn, feats)
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    out: dict[str, Any] = {"n": len(y), "n_pos": n_pos, "n_neg": n_neg,
                           "features": list(feats), **drops}
    if n_pos < 10 or n_neg < 10:
        out["error"] = (f"only {n_pos} approved and {n_neg} rejected with usable "
                        f"features — not enough to fit")
        return out

    # Standardise so coefficients are comparable to each other and the ridge
    # penalty falls evenly. Scale is stored for applying the model later.
    mu, sd = X.mean(axis=0), X.std(axis=0)
    sd = np.where(sd > 1e-9, sd, 1.0)
    Z = (X - mu) / sd

    inc = incumbent_scores(rows)
    grp = np.array([r.get("ride_id") or r.get("content_hash") for r in rows])
    fold_idx = _folds(y, folds, seed)
    fit_auc, inc_auc, coefs = [], [], []
    # Out-of-fold predictions collected across every fold, then scored once.
    # Computing a within-ride AUC per fold and averaging looked equivalent and
    # is not: each fold holds a fifth of the data, and only rides with both an
    # approved and a rejected clip contribute any pairs at all, so a per-fold
    # estimate can rest on single figures. Pooling the held-out predictions
    # keeps every score out-of-sample while scoring all pairs at once.
    oof = np.full(len(y), np.nan)
    for f in fold_idx:
        mask = np.zeros(len(y), bool)
        mask[f] = True
        if y[mask].sum() == 0 or y[mask].sum() == mask.sum():
            continue
        w = _fit(Z[~mask], y[~mask])
        coefs.append(w[1:])
        oof[mask] = Z[mask] @ w[1:]
        fit_auc.append(_auc(Z[mask] @ w[1:], y[mask]))
        if np.isfinite(inc[mask]).all():
            inc_auc.append(_auc(inc[mask], y[mask]))

    ok = np.isfinite(oof)
    w_auc, w_pairs = (_auc_within(oof[ok], y[ok], grp[ok]) if ok.any()
                      else (float("nan"), 0))
    inc_w, inc_w_pairs = _auc_within(inc[np.isfinite(inc)], y[np.isfinite(inc)],
                                     grp[np.isfinite(inc)])

    FEATS = feats
    C = np.array(coefs)
    out.update({
        "cv_auc_fit": float(np.nanmean(fit_auc)),
        "cv_auc_fit_sd": float(np.nanstd(fit_auc)),
        "cv_auc_incumbent": float(np.nanmean(inc_auc)) if inc_auc else float("nan"),
        "cv_auc_within": w_auc,
        "within_pairs": w_pairs,
        "incumbent_within": inc_w,
        "coef_mean": C.mean(axis=0).tolist(),
        "coef_sd": C.std(axis=0).tolist(),
        "coef_sign_stable": [bool(np.all(np.sign(C[:, i]) == np.sign(C[0, i])))
                             for i in range(C.shape[1])],
        "full": _fit(Z, y)[1:].tolist(),
        "mu": mu.tolist(), "sd": sd.tolist(),
    })

    # Suggested weights: positive coefficients, normalised. A negative
    # coefficient means that feature predicts *rejection*, which cannot be
    # expressed as a weight at all — it is reported, not silently clipped.
    full = np.array(out["full"])
    pos = np.clip(full, 0, None)
    out["suggested"] = {f: float(w / pos.sum()) if pos.sum() > 0 else 0.0
                        for f, w in zip(FEATS, pos)}
    out["negative"] = [f for f, w in zip(FEATS, full) if w < 0]

    # A verdict, because a fit always produces numbers and the numbers always
    # look confident. On data with no relationship at all this still emitted a
    # tidy-looking weight vector; the AUC is what says whether to believe it.
    a_fit, a_inc, spread = (out["cv_auc_fit"], out["cv_auc_incumbent"],
                            out["cv_auc_fit_sd"])
    a_win = out["cv_auc_within"]
    unstable = [f for f, ok in zip(FEATS, out["coef_sign_stable"]) if not ok]
    # Within-ride first. A pooled AUC that is not backed by a within-ride one is
    # measuring which ride a clip came from, and re-weighting the composite
    # cannot act on that — every clip in a ride would move together.
    pairs = out["within_pairs"]
    a_inc_w = out["incumbent_within"]
    # Compare on the within-ride axis, not the pooled one. Pooled AUC is
    # dominated by between-ride variation the composite was never meant to
    # capture, and judging on it hid the actual result: the hand-set weights
    # score 0.767 within rides against the fit's 0.541.
    if (np.isfinite(a_inc_w) and np.isfinite(a_win) and pairs >= 40
            and a_inc_w > a_win + 0.05):
        out["verdict"] = "keep current"
        out["why"] = (f"within rides the current weights score {a_inc_w:.2f} and "
                      f"this fit only {a_win:.2f}, on {pairs} same-ride pairs. A "
                      f"linear model cannot express the power mean the composite "
                      f"uses — being outstanding at one feature is the whole point "
                      f"of SHARPNESS, and a weighted sum cannot say it")
    elif np.isfinite(a_win) and pairs < 40:
        out["verdict"] = "too few comparisons"
        out["why"] = (f"the within-ride check rests on only {pairs} approved/rejected "
                      f"pairs from the same ride, which is not enough to tell a real "
                      f"per-clip effect from the model recognising a ride. Review more "
                      f"clips on rides where you have already approved something")
    elif np.isfinite(a_fit) and a_fit >= 0.58 and not np.isfinite(a_win):
        # A guard that cannot run must not wave the result through. This exact
        # case defaulted to "use it" the first time it was tried.
        out["verdict"] = "unverifiable"
        out["why"] = (f"pooled AUC {a_fit:.2f} looks useful, but no ride has both "
                      f"an approved and a rejected clip, so the within-ride check "
                      f"cannot run and the result cannot be told apart from the "
                      f"model simply recognising which ride a clip came from")
    elif np.isfinite(a_win) and a_win < 0.56 and np.isfinite(a_fit) and a_fit >= 0.58:
        out["verdict"] = "confounded"
        out["why"] = (f"pooled AUC {a_fit:.2f} looks useful but within-ride AUC is "
                      f"only {a_win:.2f} — the model is identifying which ride a "
                      f"clip came from, not which clip in a ride you prefer")
    elif not np.isfinite(a_fit) or a_fit < 0.58:
        out["verdict"] = "no signal"
        out["why"] = (f"held-out AUC {a_fit:.2f} is not meaningfully above 0.5 — "
                      f"these four features do not predict your decisions")
    elif np.isfinite(a_inc) and a_fit - a_inc < max(0.03, spread / 2):
        out["verdict"] = "no better"
        out["why"] = (f"held-out AUC {a_fit:.2f} vs the current weights' "
                      f"{a_inc:.2f} — inside the noise, so there is nothing to gain")
    else:
        out["verdict"] = "use it"
        out["why"] = (f"held-out AUC {a_fit:.2f} vs {a_inc:.2f} for the weights "
                      f"set by feel")
    if unstable:
        out["why"] += (f". {', '.join(unstable)} flipped sign between folds and "
                       f"should be treated as unmeasured")
    return out
