"""Forecast GTA VI's launch-day Twitch viewers hour by hour.

Reads one case (schema makefun/twitch-ccu-case/v1): the first hour's print
plus the launch-day curves of past big game launches ("comparables"), and
writes five quantiles (5/25/50/75/95%) of each of hours 2..24.

Method: two log-space regressions per hour, blended in log space,
plus a nearest-neighbour shape component, all fitted at run time from
the case's own comparables.
  1. Full fit: ln(viewers_h) ~ ln(hour-1 print) + ln(pre-release baseline)
     over every usable comparable. The hour-1 print encodes how much of
     the rolling-midnight unlock has already happened (a small hour 1
     means most timezones unlock later, so the day ramps hard); the
     pre-release baseline (median hourly viewers 72h..13h before the
     release day) measures launch hype. Residual quantiles give honest
     spreads.
  2. Console-only fit: ln(viewers_h) ~ ln(hour-1 print) over the
     comparables whose platforms name no PC (console launches, the same
     regime as GTA VI: rolling midnight, no global PC unlock). Its
     slope near 0.7 extrapolates far better than the full fit's flat
     late-hour slopes when the hour-1 print exceeds every comparable.
     Each hour's slope is shrunk 30% toward the pooled slope over all
     hours (SHRINK_SUB), which regularises the small 8-curve console
     fit and improves held-out large launches.
  3. Neighbour shapes: the 6 comparables nearest the target in
     (ln hour-1 print, ln baseline), kernel-weighted, contribute their
     mean log-ratio profile and its residual spread. This local
     component adapts the forecast to launches shaped like the target
     instead of forcing one global line through all regimes.
The full fit carries weight 0.25 against the console-only fit, and the
neighbour component carries weight 0.3 against the regression blend.
The final log-quantile offsets from the median are scaled by 0.9
(SPREAD_SCALE): in-sample residual quantiles from the small console
fit overstate the spread, and the calibrated multiplier brings the
leave-one-comparable-out 90% coverage toward its nominal level.
All four constants (blend weight, neighbour weight, shrinkage,
spread scale) were chosen by leave-one-comparable-out validation over
the comparables (never by tuning on the practice answers): the blend
beats the regression-only model on held-out console launches and, by
a wide margin, when the largest launches are held out together.

Fitted parameters are computed at run time from the case's own
comparables (the final case carries 42, practice cases 38). The
FALLBACK_* tables below hold the same fits over the 37 practice
comparables and are used only if a case carries too few usable curves.
Any failure anywhere falls back to a persistence forecast (every hour
at the hour-1 print, spread widening like a random walk), so the
program never crashes and always writes a valid forecast.

Standard library only, deterministic (sorted iteration, no randomness),
offline, well under the 8-second limit.
"""

from __future__ import annotations

import json
import math
import os
from datetime import datetime
from pathlib import Path
from statistics import NormalDist, pstdev

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
NORMAL = NormalDist()
# Persistence fallback: log-spread per square-root hour.
SIGMA = 0.15
# Blend weight of the full fit (1 - weight goes to the console-only fit).
BLEND_W = 0.25
# Console-fit slope shrinkage toward the pooled all-hours slope, and
# the final spread-scale multiplier on log-quantile offsets from the
# median. Both calibrated by leave-one-comparable-out validation.
SHRINK_SUB = 0.3
SPREAD_SCALE = 0.9
# Neighbour-shape component: nearest KNN_K curves, Gaussian kernel
# bandwidth KNN_H in standardised (ln p1, ln baseline) distance, weight
# KNN_W against the regression blend. Needs at least MIN_KNN_CURVES
# curves carrying the hour.
KNN_K = 6
KNN_H = 1.0
KNN_W = 0.3
MIN_KNN_CURVES = 3
# A comparable counts as usable with an hour-1 print and this many of
# hours 2..24 present.
MIN_DAY_HOURS = 12
MIN_FULL_CURVES = 6
MIN_SUB_CURVES = 4
MIN_FIT_POINTS = 3

# Fallback fits over the 37 practice comparables, per hour:
#   FULL: (a, b1, b2, [residual quantiles at LEVELS])
#   SUBSET: (a, b, [residual quantiles at LEVELS])
# Used only when the runtime case carries too few usable comparables.
FALLBACK_FULL = {
    2: (-0.3049, 1.0251, -0.0060, [-0.3021, -0.1557, -0.0824, 0.075, 0.4764]),
    3: (-0.2542, 1.0008, 0.0029, [-0.7747, -0.2704, -0.0632, 0.1493, 0.6118]),
    4: (0.1556, 0.9636, -0.0059, [-0.6425, -0.4019, -0.0297, 0.1494, 0.6109]),
    5: (0.1468, 0.9940, -0.0493, [-0.8927, -0.2832, -0.0617, 0.2847, 0.6613]),
    6: (0.3109, 0.9936, -0.0634, [-1.0753, -0.4277, -0.0441, 0.3257, 0.9423]),
    7: (1.0749, 0.9211, -0.0532, [-1.0296, -0.4629, -0.036, 0.3491, 1.0836]),
    8: (1.8356, 0.8585, -0.0542, [-0.9735, -0.4803, -0.0498, 0.2969, 1.1749]),
    9: (3.3959, 0.7351, -0.0753, [-0.9, -0.4226, -0.0541, 0.3415, 1.2668]),
    10: (5.3075, 0.5832, -0.0960, [-1.0804, -0.456, -0.1044, 0.3573, 1.6664]),
    11: (5.8298, 0.5427, -0.0989, [-1.0965, -0.4789, -0.1879, 0.3229, 1.7475]),
    12: (6.3156, 0.5093, -0.1034, [-1.1577, -0.512, -0.1314, 0.3559, 1.5807]),
    13: (7.2797, 0.4361, -0.0982, [-1.1555, -0.4801, -0.0361, 0.4247, 1.5042]),
    14: (8.2757, 0.3672, -0.1079, [-1.2447, -0.571, 0.0093, 0.7342, 1.5184]),
    15: (9.3245, 0.2923, -0.1152, [-1.2025, -0.6894, -0.0512, 0.7415, 1.6093]),
    16: (10.3772, 0.2079, -0.1091, [-1.1121, -0.7003, -0.0735, 0.7394, 1.4939]),
    17: (10.7976, 0.1814, -0.1113, [-1.1335, -0.6917, -0.1004, 0.8207, 1.5099]),
    18: (11.2279, 0.1433, -0.0978, [-1.0448, -0.7633, -0.1404, 0.805, 1.377]),
    19: (11.5080, 0.1147, -0.0779, [-1.0521, -0.5941, -0.0469, 0.6509, 1.1878]),
    20: (11.5989, 0.1081, -0.0768, [-0.986, -0.6548, -0.08, 0.5136, 1.0903]),
    21: (11.6451, 0.1097, -0.0838, [-0.9835, -0.6743, -0.0506, 0.6238, 1.06]),
    22: (11.5366, 0.1153, -0.0821, [-0.9792, -0.6751, -0.0132, 0.6215, 1.0333]),
    23: (11.2685, 0.1323, -0.0811, [-1.0616, -0.5444, 0.0229, 0.609, 0.9895]),
    24: (11.0515, 0.1484, -0.0932, [-0.9931, -0.5798, 0.068, 0.5969, 1.0764]),
}
FALLBACK_SUB = {
    2: (0.5241, 0.9334, [-0.1685, -0.0811, -0.0138, 0.0703, 0.1969]),
    3: (0.0099, 0.9529, [-0.5636, -0.0899, 0.1281, 0.1598, 0.3406]),
    4: (0.6078, 0.8958, [-0.3405, -0.0950, -0.0273, 0.1928, 0.2905]),
    5: (-0.0307, 0.9853, [-0.2554, -0.0720, -0.0128, 0.1164, 0.2318]),
    6: (0.4415, 0.9516, [-0.2695, -0.1791, 0.0055, 0.1948, 0.2512]),
    7: (1.1086, 0.8879, [-0.3402, -0.1555, 0.0436, 0.1568, 0.2847]),
    8: (1.4733, 0.8524, [-0.3942, -0.1328, 0.0719, 0.1651, 0.2729]),
    9: (1.8201, 0.8210, [-0.2684, -0.2142, -0.0197, 0.1957, 0.3049]),
    10: (1.6500, 0.8356, [-0.2629, -0.1919, -0.0874, 0.1714, 0.3899]),
    11: (2.1378, 0.7977, [-0.3492, -0.2225, 0.0123, 0.1507, 0.3965]),
    12: (2.1351, 0.8081, [-0.3049, -0.1638, -0.0441, 0.1074, 0.4113]),
    13: (2.7235, 0.7729, [-0.3481, -0.2193, -0.0347, 0.2327, 0.3837]),
    14: (2.9452, 0.7679, [-0.3341, -0.2671, -0.0189, 0.2486, 0.3783]),
    15: (3.5900, 0.7282, [-0.3580, -0.1852, -0.0039, 0.1903, 0.3653]),
    16: (3.8789, 0.7104, [-0.3705, -0.1574, 0.0582, 0.1288, 0.3239]),
    17: (3.5357, 0.7435, [-0.4232, -0.1447, 0.0473, 0.1459, 0.3678]),
    18: (3.8287, 0.7245, [-0.4558, -0.1533, 0.0198, 0.2675, 0.3379]),
    19: (3.9239, 0.7208, [-0.4731, -0.1738, 0.1174, 0.2041, 0.3116]),
    20: (3.6996, 0.7437, [-0.4190, -0.1451, 0.1125, 0.1654, 0.2560]),
    21: (3.7532, 0.7383, [-0.4213, -0.1122, 0.1107, 0.1663, 0.2265]),
    22: (3.8604, 0.7239, [-0.4450, -0.1109, 0.1236, 0.1827, 0.2198]),
    23: (3.1501, 0.7772, [-0.5290, -0.1308, 0.1197, 0.2271, 0.2805]),
    24: (2.7527, 0.8041, [-0.5601, -0.1529, 0.0790, 0.2155, 0.4098]),
}


def _parse_time(text):
    """Parse an ISO-8601 UTC timestamp; None when unparseable."""
    if not isinstance(text, str):
        return None
    text = text.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _quantile(sorted_vals, tau):
    """Linear-interpolation quantile of an ascending list."""
    n = len(sorted_vals)
    if n == 0:
        return 0.0
    if n == 1:
        return sorted_vals[0]
    tau = min(max(tau, 0.0), 1.0)
    pos = tau * (n - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


def _median(values):
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        return None
    mid = n // 2
    if n % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def _ols1(rows):
    """Least squares y ~ a + b*x; rows [(x, y)]. None when degenerate."""
    n = len(rows)
    if n < MIN_FIT_POINTS:
        return None
    mx = sum(r[0] for r in rows) / n
    my = sum(r[1] for r in rows) / n
    sxx = sum((r[0] - mx) ** 2 for r in rows)
    if sxx <= 0:
        return None
    sxy = sum((r[0] - mx) * (r[1] - my) for r in rows)
    b = sxy / sxx
    a = my - b * mx
    res = sorted(r[1] - (a + b * r[0]) for r in rows)
    return (a, b, res)


def _ols2(rows):
    """Least squares y ~ a + b1*x1 + b2*x2; rows [(x1, x2, y)]."""
    n = len(rows)
    if n < MIN_FIT_POINTS:
        return None
    mx1 = sum(r[0] for r in rows) / n
    mx2 = sum(r[1] for r in rows) / n
    my = sum(r[2] for r in rows) / n
    s11 = sum((r[0] - mx1) ** 2 for r in rows)
    s22 = sum((r[1] - mx2) ** 2 for r in rows)
    s12 = sum((r[0] - mx1) * (r[1] - mx2) for r in rows)
    s1y = sum((r[0] - mx1) * (r[2] - my) for r in rows)
    s2y = sum((r[1] - mx2) * (r[2] - my) for r in rows)
    det = s11 * s22 - s12 * s12
    if abs(det) < 1e-12:
        return None
    b1 = (s22 * s1y - s12 * s2y) / det
    b2 = (s11 * s2y - s12 * s1y) / det
    a = my - b1 * mx1 - b2 * mx2
    res = sorted(r[2] - (a + b1 * r[0] + b2 * r[1]) for r in rows)
    return (a, b1, b2, res)


def _number(value):
    """A finite float for value, or None."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and math.isfinite(value):
        return float(value)
    return None


def _release_day(hourly, day_start):
    """Map release-day hour (1..24) to mean viewers from hourly points."""
    start = _parse_time(day_start)
    if start is None or not isinstance(hourly, list):
        return {}
    buckets = {}
    for pt in hourly:
        if not isinstance(pt, dict):
            continue
        at = _parse_time(pt.get("from"))
        viewers = _number(pt.get("viewers"))
        if at is None or viewers is None or viewers <= 0:
            continue
        idx = int((at - start).total_seconds() // 3600)
        if 0 <= idx < 24:
            buckets.setdefault(idx + 1, []).append(viewers)
    return {h: sum(v) / len(v) for h, v in buckets.items()}


def _baseline(points, day_start):
    """Median hourly viewers 72h..13h before the release day, or None.

    `points` is a list of (timestamp, viewers) pairs. The window ends
    before the New Zealand midnight unlock (~13h before UTC midnight),
    so it measures pre-release hype, uncontaminated by unlocks.
    """
    start = _parse_time(day_start)
    if start is None:
        return None
    vals = []
    for at_text, viewers in points:
        at = _parse_time(at_text)
        if at is None or viewers is None or viewers <= 0:
            continue
        dt = (at - start).total_seconds() / 3600.0
        if -72.0 <= dt < -13.0:
            vals.append(viewers)
    return _median(vals)


def _is_console_only(platforms):
    text = (platforms or "").lower()
    return "pc" not in text and "stadia" not in text


def _curves(case):
    """Usable comparable curves, sorted by name for determinism."""
    raw = case.get("comparables")
    if not isinstance(raw, list):
        return []
    curves = []
    for comp in raw:
        if not isinstance(comp, dict):
            continue
        day = _release_day(comp.get("hourly"), comp.get("day_start"))
        p1 = day.get(1)
        if p1 is None or p1 <= 0:
            continue
        if sum(1 for h in range(2, 25) if h in day) < MIN_DAY_HOURS:
            continue
        hourly = comp.get("hourly") or []
        pts = [(p.get("from"), _number(p.get("viewers")))
               for p in hourly if isinstance(p, dict)]
        base = _baseline(pts, comp.get("day_start"))
        curves.append({
            "name": str(comp.get("name", "")),
            "p1": p1,
            "day": day,
            "base": base,
            "console": _is_console_only(comp.get("platforms")),
        })
    curves.sort(key=lambda c: (c["name"], c["p1"]))
    return curves


def _target_print(case):
    """The case's hour-1 print, falling back to the last known level."""
    known = case.get("known")
    if isinstance(known, list):
        for row in known:
            if isinstance(row, dict) and row.get("hour") == 1:
                v = _number(row.get("viewers"))
                if v is not None and v > 0:
                    return v
    hourly = case.get("history_hourly")
    if isinstance(hourly, list) and hourly:
        for row in reversed(hourly):
            if isinstance(row, dict):
                v = _number(row.get("viewers"))
                if v is not None and v > 0:
                    return v
    return 1.0


def _target_baseline(case):
    """Pre-release baseline for the target case, or None."""
    day_start = case.get("day_start")
    hourly = case.get("history_hourly")
    if isinstance(hourly, list) and hourly:
        pts = [(p.get("from"), _number(p.get("viewers")))
               for p in hourly if isinstance(p, dict)]
        base = _baseline(pts, day_start)
        if base is not None:
            return base
    # Fall back to bucketing the raw history points into whole hours.
    history = case.get("history")
    if isinstance(history, list) and history:
        start = _parse_time(day_start)
        if start is None:
            return None
        buckets = {}
        for pt in history:
            if not isinstance(pt, dict):
                continue
            at = _parse_time(pt.get("at"))
            v = _number(pt.get("viewers"))
            if at is None or v is None or v <= 0:
                continue
            idx = int((at - start).total_seconds() // 3600)
            buckets.setdefault(idx, []).append(v)
        vals = []
        for idx in sorted(buckets):
            if -72 <= idx < -13:
                vals.extend(buckets[idx])
        return _median(vals)
    return None


def _fit_full(curves):
    """Per-hour full fits: {hour: ('two', a, b1, b2, res) | ('one', ...)}."""
    fits = {}
    if len(curves) < MIN_FULL_CURVES:
        return fits
    for h in range(2, 25):
        rows2 = [(math.log(c["p1"]), math.log(c["base"]), math.log(c["day"][h]))
                 for c in curves
                 if h in c["day"] and c["day"][h] > 0 and c["base"]]
        fit = _ols2(rows2) if len(rows2) >= MIN_FIT_POINTS else None
        if fit is not None:
            fits[h] = ("two",) + fit
            continue
        rows1 = [(math.log(c["p1"]), math.log(c["day"][h]))
                 for c in curves if h in c["day"] and c["day"][h] > 0]
        fit1 = _ols1(rows1)
        if fit1 is not None:
            fits[h] = ("one",) + fit1
    return fits


def _fit_subset(curves):
    """Per-hour console-only fits: {hour: (a, b, res)}.

    Each hour's slope is shrunk SHRINK_SUB toward the pooled slope
    fitted over all hours at once, which regularises the small
    console-only sample; the intercept is refit through the hour's
    mean point and the residuals recomputed from the shrunk line.
    """
    use = [c for c in curves if c["console"]]
    fits = {}
    if len(use) < MIN_SUB_CURVES:
        return fits
    pooled = None
    rows_all = [(math.log(c["p1"]), math.log(c["day"][h]))
                for c in use for h in range(2, 25)
                if h in c["day"] and c["day"][h] > 0]
    pooled_fit = _ols1(rows_all)
    if pooled_fit is not None:
        pooled = pooled_fit[1]
    for h in range(2, 25):
        rows = [(math.log(c["p1"]), math.log(c["day"][h]))
                for c in use if h in c["day"] and c["day"][h] > 0]
        fit = _ols1(rows)
        if fit is None:
            continue
        a, b, res = fit
        if pooled is not None and SHRINK_SUB > 0.0:
            n = len(rows)
            mx = sum(r[0] for r in rows) / n
            my = sum(r[1] for r in rows) / n
            b = (1.0 - SHRINK_SUB) * b + SHRINK_SUB * pooled
            a = my - b * mx
            res = sorted(r[1] - (a + b * r[0]) for r in rows)
        fits[h] = (a, b, res)
    return fits


def _exp_capped(log_value):
    try:
        value = math.exp(log_value)
    except OverflowError:
        return 1e9
    if not math.isfinite(value):
        return 1e9
    return value


def _log_q_from_fit(fit, x, xb, levels):
    """Log-space quantiles from a fit: [center + residual quantile]."""
    if fit[0] == "two":
        _, a, b1, b2, res = fit
        center = a + b1 * x + b2 * xb
    else:
        _, a, b, res = fit
        center = a + b * x
    return [center + _quantile(res, tau) for tau in levels]


def _log_q_embedded(table_full, table_sub, hour, x, xb, levels):
    """Log-space quantiles from the embedded fallback tables; None if unusable."""
    full_lq = None
    sub_lq = None
    if hour in table_full and xb is not None:
        a, b1, b2, offs = table_full[hour]
        center = a + b1 * x + b2 * xb
        full_lq = [center + _quantile(sorted(offs), tau) for tau in levels]
    if hour in table_sub:
        a, b, offs = table_sub[hour]
        center = a + b * x
        sub_lq = [center + _quantile(sorted(offs), tau) for tau in levels]
    if full_lq is not None and sub_lq is not None:
        return [BLEND_W * f + (1.0 - BLEND_W) * s
                for f, s in zip(full_lq, sub_lq)]
    return full_lq if full_lq is not None else sub_lq


def _knn_scales(curves):
    """Standard deviations of ln(p1) and ln(baseline) over the curves.

    None when too few curves to run the neighbour component.
    """
    if len(curves) < MIN_KNN_CURVES:
        return None
    sd_x = pstdev([math.log(c["p1"]) for c in curves]) or 1.0
    xb = [math.log(c["base"]) for c in curves if c["base"]]
    sd_b = (pstdev(xb) or 1.0) if xb else 1.0
    return (sd_x, sd_b)


def _log_q_knn(curves, scales, x, xb, hour, levels):
    """Log-space quantiles from the nearest-neighbour shape component.

    The KNN_K curves nearest the target in standardised (ln p1,
    ln baseline) distance contribute their kernel-weighted mean
    log-ratio for this hour, with spreads from the neighbours'
    residual quantiles around that mean. None when too few curves
    carry the hour.
    """
    if scales is None:
        return None
    sd_x, sd_b = scales
    scored = []
    for c in curves:
        if hour not in c["day"] or c["day"][hour] <= 0:
            continue
        d = abs(math.log(c["p1"]) - x) / sd_x
        if xb is not None and c["base"]:
            d += abs(math.log(c["base"]) - xb) / sd_b
        scored.append((d, c))
    if len(scored) < MIN_KNN_CURVES:
        return None
    scored.sort(key=lambda t: t[0])
    use = scored[:KNN_K]
    wts = [math.exp(-(d * d) / (2.0 * KNN_H * KNN_H)) for d, _ in use]
    wsum = sum(wts)
    if wsum <= 0:
        return None
    ratios = [math.log(c["day"][hour] / c["p1"]) for _, c in use]
    mean_r = sum(w * r for w, r in zip(wts, ratios)) / wsum
    res = sorted(r - mean_r for r in ratios)
    center = x + mean_r
    return [center + _quantile(res, tau) for tau in levels]


def _persistence(p1, hour, levels):
    spread = SIGMA * math.sqrt(hour - 1)
    return [_exp_capped(math.log(max(1.0, p1)) + NORMAL.inv_cdf(tau) * spread)
            for tau in levels]


def _clean(values):
    """Round half-up to 2 decimals, clip, enforce non-decreasing."""
    out = []
    prev = 1.0
    for v in values:
        if not math.isfinite(v):
            v = 1e9 if v > 0 else 1.0
        r = math.floor(v * 100.0 + 0.5) / 100.0
        r = min(max(r, 1.0), 1e9)
        if r < prev:
            r = prev
        prev = r
        out.append(int(r) if float(r).is_integer() else r)
    return out


def _levels_of(case):
    raw = case.get("levels")
    if isinstance(raw, list) and len(raw) >= 2:
        vals = [_number(v) for v in raw]
        if all(v is not None and 0.0 < v < 1.0 for v in vals):
            return vals
    return list(LEVELS)


def _hours_of(case):
    rows = case.get("forecast")
    hours = []
    if isinstance(rows, list):
        for row in rows:
            if isinstance(row, dict) and isinstance(row.get("hour"), int):
                hours.append(row["hour"])
    hours = sorted(set(h for h in hours if 2 <= h <= 24))
    return hours if len(hours) == 23 else list(range(2, 25))


def forecast_case(case):
    """Build the forecast dict for one case; never raises."""
    try:
        return _forecast_case_inner(case)
    except Exception:
        p1 = 1.0
        try:
            p1 = _target_print(case)
        except Exception:
            pass
        return {"hours": [{"hour": h, "q": _clean(_persistence(p1, h, LEVELS))}
                          for h in range(2, 25)]}


def _forecast_case_inner(case):
    levels = _levels_of(case)
    hours = _hours_of(case)
    p1 = max(1.0, _target_print(case))
    base = _target_baseline(case)
    x = math.log(p1)
    xb = math.log(base) if base else None
    curves = _curves(case)
    full_fits = _fit_full(curves)
    sub_fits = _fit_subset(curves)
    knn_scales = _knn_scales(curves)
    med_idx = min(range(len(levels)), key=lambda i: abs(levels[i] - 0.5))
    rows = []
    for h in hours:
        full_lq = None
        sub_lq = None
        if h in full_fits:
            fit = full_fits[h]
            if not (fit[0] == "two" and xb is None):
                full_lq = _log_q_from_fit(
                    fit, x, xb if xb is not None else 0.0, levels)
        if h in sub_fits:
            a, b, res = sub_fits[h]
            sub_lq = [a + b * x + _quantile(res, tau) for tau in levels]
        if full_lq is not None and sub_lq is not None:
            reg = [BLEND_W * f + (1.0 - BLEND_W) * s
                   for f, s in zip(full_lq, sub_lq)]
        elif full_lq is not None:
            reg = full_lq
        elif sub_lq is not None:
            reg = sub_lq
        else:
            reg = _log_q_embedded(
                FALLBACK_FULL, FALLBACK_SUB, h, x, xb, levels)
        knn_lq = _log_q_knn(curves, knn_scales, x, xb, h, levels)
        if reg is not None and knn_lq is not None:
            log_q = [(1.0 - KNN_W) * r + KNN_W * k
                     for r, k in zip(reg, knn_lq)]
        elif reg is not None:
            log_q = reg
        elif knn_lq is not None:
            log_q = knn_lq
        else:
            spread = SIGMA * math.sqrt(h - 1)
            log_q = [math.log(max(1.0, p1)) + NORMAL.inv_cdf(tau) * spread
                     for tau in levels]
        med = log_q[med_idx]
        log_q = [med + (v - med) * SPREAD_SCALE for v in log_q]
        rows.append({"hour": h, "q": _clean([_exp_capped(v) for v in log_q])})
    return {"hours": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        try:
            case = json.loads(path.read_text(encoding="utf-8"))
            forecast = forecast_case(case)
        except Exception:
            forecast = {"hours": [{"hour": h, "q": _clean(_persistence(1.0, h, LEVELS))}
                                  for h in range(2, 25)]}
        out_path = OUTPUT_DIR / (path.stem + ".out")
        out_path.write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
