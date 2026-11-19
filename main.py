"""GTA VI trophy rarity forecast: per-trophy trend forecaster.

Reads each case file (<id>.in) from $MAKEFUN_INPUT_DIR (else /input) and writes
<id>.out to $MAKEFUN_OUTPUT_DIR (else /output): for every trophy, five
quantiles (0.05, 0.25, 0.5, 0.75, 0.95) of its Exophase earn rate at the case
target, in percent.

Method, per trophy:
  * Build the series of past reads (time, players tracked, earn rate).
  * Fit a weighted least-squares line of logit(rate) on ln(1 + hours since
    the game's release): trophy progress saturates, so log-time
    extrapolates gently where a straight line in time would run away.
    Which window of history to fit (last 12h, 24h, 48h, or all of it) is
    chosen by backtesting inside the case itself: fit on the early part
    of the window, predict the later reads, keep the window whose
    predictions were closest.
  * Slopes are shrunk toward the median slope across the game's trophies
    (empirical-Bayes style): a trophy whose own series is noisy borrows
    strength from the rest of the game, a clean series stands on its own.
  * Spread is honest and measured, not assumed: the same within-case
    backtest gives the size of a real prediction error at a known horizon,
    scaled to the target horizon, plus the fit's residual noise. A trend
    that has predicted well gets a narrow band; a noisy one gets a wide one.
  * When the whole page is stale -- every read identical, players tracked
    unchanged, so Exophase has not refreshed once in the history -- the
    trend says "no change" but the next refresh will dilute every rate
    with a flood of fresh players (launch night's first refresh took the
    tracked base from 638 to 3868 players while the new players had earned
    almost nothing). For a short horizon that case gets a dilution prior
    instead of persistence, with a wide spread; for a long horizon it
    keeps the last rate with a very wide spread, since progression then
    matters too.
  * The one trophy every tracked player holds (rate 100%) stays near 100%:
    every new player earns it at once.

Python 3.13, standard library only. Deterministic: no randomness, no time
or network reads, so two runs on one case write identical bytes.
"""

import json
import math
import os
from datetime import datetime
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# A fresh cohort's chance of holding a trophy, relative to the rate among
# the players tracked so far (calibrated on launch night's first refresh:
# 638 -> 3868 tracked, 943 -> 4276 trophies earned, the universal trophy
# held by every new player, the rest of the new players' 103 trophies
# spread over the other 54). The refresh's player growth at that moment.
FRESH_RELATIVE_RATE = 0.0667
REFRESH_GROWTH = 5.0
STALE_SPREAD = 1.0
# Logit uncertainty per unit of log-time extrapolated past the last read:
# within-case backtests cannot see regime change beyond the data, so the
# spread also grows with the extrapolation distance itself.
HORIZON_SLOPE = 0.75
# Candidate fit windows, in hours (None = all history).
WINDOWS = [12.0, 24.0, 48.0, None]


def clamp_pct(x):
    return min(99.99, max(0.01, x))


def logit_pct(x):
    p = clamp_pct(x) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit_pct(y):
    if y > 20.0:
        return 99.99
    if y < -20.0:
        return 0.01
    return 100.0 / (1.0 + math.exp(-y))


def round4(x):
    """Round to 4 decimals, half up, returning an int for whole numbers."""
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if float(value).is_integer() else value


def parse_time(text):
    """ISO 8601 timestamp (or plain date) -> seconds since the epoch."""
    if not text:
        return None
    txt = str(text).strip()
    if len(txt) == 10:
        txt += "T00:00:00Z"
    return datetime.fromisoformat(txt.replace("Z", "+00:00")).timestamp()


def wls(points):
    """Weighted least squares y ~ a + b*x.

    points: list of (x, y, w). Returns (a, b, resid_sd, se_b, xbar, sxx)
    or None when the fit is degenerate.
    """
    n = len(points)
    if n == 0:
        return None
    sw = sum(w for _, _, w in points)
    if sw <= 0:
        return None
    xbar = sum(w * x for x, _, w in points) / sw
    ybar = sum(w * y for _, y, w in points) / sw
    sxx = sum(w * (x - xbar) ** 2 for x, _, w in points)
    if sxx <= 1e-12 or n < 2:
        return (ybar, 0.0, 0.0, 0.0, xbar, sxx)
    sxy = sum(w * (x - xbar) * (y - ybar) for x, y, w in points)
    b = sxy / sxx
    a = ybar - b * xbar
    if n > 2:
        resid = math.sqrt(sum((y - (a + b * x)) ** 2 for x, y, w in points) / (n - 2))
    else:
        resid = 0.0
    se_b = resid / math.sqrt(sxx) if sxx > 0 else 0.0
    return (a, b, resid, se_b, xbar, sxx)


def _points(series, t_last, window_hours):
    """Log-time points for one trophy over a window ending at the last read."""
    pts = []
    for t_h, _n_pl, rate in series:
        if window_hours is not None and t_last - t_h > window_hours:
            continue
        w = 0.5 ** ((t_last - t_h) / 24.0)
        pts.append((math.log1p(t_h), logit_pct(rate), w))
    return pts


def _cv_error(series, window_hours):
    """Backtest error of the log-time trend over one window.

    Every window is judged on the same validation segment -- the last
    V hours of the history -- fitted on the W hours before it, so a short
    window cannot win just by having a shorter validation horizon. Falls
    back to a 65/35 split when the history is too short to split that way.
    Returns (mean absolute logit error, validation span in log-time) or None.
    """
    t_last = series[-1][0]
    span_h = t_last - series[0][0]
    pts_all = _points(series, t_last, None)
    if len(pts_all) < 4:
        return None
    v_h = 12.0 if span_h >= 48.0 else max(span_h / 4.0, 1.0)
    t_cut = t_last - v_h
    fit_pts = []
    val_pts = []
    for x, y, w in pts_all:
        t_h = math.expm1(x)
        if t_h <= t_cut and (window_hours is None or t_h > t_cut - window_hours):
            fit_pts.append((x, y, w))
        elif t_h > t_cut:
            val_pts.append((x, y, w))
    if len(fit_pts) < 3 or not val_pts:
        # Fallback: 65/35 split of the window's points.
        pts = _points(series, t_last, window_hours)
        if len(pts) < 4:
            return None
        xs = [p[0] for p in pts]
        x_split = xs[0] + 0.65 * (xs[-1] - xs[0])
        early = [p for p in pts if p[0] <= x_split]
        late = [p for p in pts if p[0] > x_split]
        if len(early) < 3 or not late:
            return None
        fit = wls(early)
        if fit is None:
            return None
        a, b = fit[0], fit[1]
        errs = [abs(y - (a + b * x)) for x, y, _w in late]
        return sum(errs) / len(errs), xs[-1] - x_split
    fit = wls(fit_pts)
    if fit is None:
        return None
    a, b = fit[0], fit[1]
    errs = [abs(y - (a + b * x)) for x, y, _w in val_pts]
    return sum(errs) / len(errs), math.log1p(t_last) - math.log1p(t_cut)


def forecast_trophy(series, x_target, horizon_days, horizon_hours, slope_median):
    """series: [(t_hours, players, rate_pct)] oldest first, non-null rates.
    Returns (median_logit, spread_logit)."""
    last_rate = series[-1][2]
    y_last = logit_pct(last_rate)

    # The trophy everybody holds stays held by everybody: every new
    # player earns it at once, so it sits at the top of the scale.
    if last_rate >= 99.9:
        return logit_pct(99.99), 0.15

    if len(series) < 3:
        return y_last, 1.0

    t_last = series[-1][0]
    all_rates = [s[2] for s in series]
    all_n = [s[1] for s in series]
    stale = (max(all_rates) - min(all_rates) < 0.02
             and max(all_n) == min(all_n))

    if stale and horizon_hours <= 6.0:
        # Dilution prior: one refresh adds REFRESH_GROWTH times as many
        # players, each holding this trophy with probability
        # FRESH_RELATIVE_RATE times the current rate.
        r = last_rate / 100.0
        q = FRESH_RELATIVE_RATE * r
        diluted = (r + q * REFRESH_GROWTH) / (1.0 + REFRESH_GROWTH)
        return logit_pct(diluted * 100.0), STALE_SPREAD

    # Pick the fit window by within-case backtest; prefer the longer
    # window when its error is within 5% of the best.
    best = None  # (cv_err, window_hours)
    for window in WINDOWS:
        got = _cv_error(series, window)
        if got is None:
            continue
        err, _span = got
        if best is None or err < best[0]:
            best = (err, window)
    if best is None:
        return y_last, 1.0
    cv_err, window = best
    for window2 in WINDOWS:
        if window2 is None or (window is not None and window2 <= window):
            continue
        got = _cv_error(series, window2)
        if got is not None and got[0] <= best[0] * 1.05:
            cv_err, window = got[0], window2

    pts = _points(series, t_last, window)
    fit = wls(pts)
    if fit is None:
        return y_last, 1.0
    a, b, resid, se_b, xbar, sxx = fit

    # Shrink the slope toward the game's median slope, more when the
    # trophy's own slope is uncertain.
    if slope_median is not None and se_b > 0:
        prior_sd = 0.5
        w_shrink = prior_sd * prior_sd / (prior_sd * prior_sd + se_b * se_b)
        b = w_shrink * b + (1.0 - w_shrink) * slope_median
        a = (sum(w * y for _x, y, w in pts) / sum(w for _x, _y, w in pts)
             - b * xbar)
    y_med = a + b * x_target

    # Never let an extrapolation run away from the last read, and never
    # let a non-universal trophy's median touch 100%: on the logit scale
    # the last few hundredths of a percent cost whole units, and a rate
    # that high needs every brand-new player to hold the trophy too.
    y_med = min(max(y_med, y_last - 4.0), y_last + 4.0, logit_pct(99.9))

    # Spread: the backtest's error, scaled from its horizon to the target
    # horizon in log-time distance, plus residual noise, plus a term for
    # the extrapolation distance itself: a backtest inside the data cannot
    # measure how the world changes past the last read.
    got = _cv_error(series, window)
    _err, span = got if got else (cv_err, 1.0)
    x_last = math.log1p(t_last)
    scale = math.sqrt(max((x_target - (x_last - span)) / max(span, 1e-9), 1.0))
    scale = min(scale, 3.0)
    spread = math.sqrt((cv_err * scale) ** 2 + min(resid, 1.0) ** 2
                       + (HORIZON_SLOPE * (x_target - x_last)) ** 2 + 0.10 ** 2)
    if stale:
        spread = max(spread, 1.2)
    spread = min(max(spread, 0.30), 1.5)
    return y_med, spread


def forecast_case(case, name):
    if case.get("levels") != LEVELS:
        raise ValueError(f"{name} asks for levels {case.get('levels')}, not {LEVELS}")
    history = case.get("history") or []
    if not history:
        raise ValueError(f"{name} has no history")
    trophies = case["trophies"]

    release = parse_time((case.get("game") or {}).get("release_date"))
    if release is None:
        release = parse_time(history[0]["at"])
    target = parse_time(case["target"])
    horizon_minutes = float(case["horizon_minutes"])
    horizon_days = horizon_minutes / 1440.0
    x_target = math.log1p((target - release) / 3600.0)

    # Per-trophy series, then a first pass for the median slope.
    all_series = []
    for i, _trophy in enumerate(trophies):
        series = []
        for read in history:
            rate = read["rates"][i]
            if rate is None:
                continue
            t_h = (parse_time(read["at"]) - release) / 3600.0
            series.append((t_h, read["players_tracked"], float(rate)))
        all_series.append(series)

    slopes = []
    for series in all_series:
        if len(series) < 3 or series[-1][2] >= 99.9:
            continue
        pts = _points(series, series[-1][0], None)
        fit = wls(pts)
        if fit is not None:
            slopes.append(fit[1])
    slopes.sort()
    slope_median = slopes[len(slopes) // 2] if slopes else None

    rows = []
    for i, trophy in enumerate(trophies):
        series = all_series[i]
        if not series:
            # No read ever listed this trophy: near zero, wide spread.
            y_med, spread = logit_pct(0.05), 1.2
        else:
            y_med, spread = forecast_trophy(series, x_target,
                                            horizon_days, horizon_minutes / 60.0,
                                            slope_median)
        qs = [round4(clamp_pct(inv_logit_pct(y_med + z * spread))) for z in Z]
        for k in range(1, 5):
            if qs[k] < qs[k - 1]:
                qs[k] = qs[k - 1]
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
