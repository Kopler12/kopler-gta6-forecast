#!/usr/bin/env python3
"""Forecast a game's launch-day Twitch viewers, hour by hour.

For every case file in the input directory this program predicts five
quantiles (5/25/50/75/95%) of the concurrent-viewer count for hours
2..24 of release day, given only the hour-1 print.

Method: scale-weighted empirical launch curves.

  * Every past launch ("comparable") in the case is reduced to its
    release-day hourly shape, expressed as log-ratios to its own
    hour-1 audience:  r_h = ln(viewers_h / viewers_1).
  * Comparables whose hour-1 audience is close in size to this case's
    hour-1 audience receive the most weight, via a Gaussian kernel on
    the log-scale distance with bandwidth 0.8.
  * For each hour and each level, the forecast log-ratio is the
    weighted quantile of the comparables' log-ratios at that hour.
  * The five quantile curves are lightly smoothed across neighbouring
    hours, then mapped back to viewers:  q = p1 * exp(log-ratio).

With too few usable comparables the program falls back to a
persistence forecast: the hour-1 audience held flat, with a log-normal
spread that widens like sqrt(elapsed hours).

Standard library only.  Deterministic: identical inputs always give
byte-identical outputs.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

# Quantile levels of the case schema (also read from each case file).
DEFAULT_LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]

# Standard-normal quantiles at DEFAULT_LEVELS, for the fallback.
Z_SCORES = [-1.645, -0.674, 0.0, 0.674, 1.645]

# Gaussian kernel bandwidth on the log scale-distance between a
# comparable's hour-1 audience and this case's hour-1 audience.
KERNEL_BANDWIDTH = 0.8

# A comparable is usable when its release day has an hour-1 print and
# at least this many of hours 1..24.
MIN_DAY_HOURS = 12

# Below this many usable comparables the empirical curves are not
# trusted and the persistence fallback is used instead.
MIN_COMPARABLES = 5

# Per-hour log spread of the persistence fallback at sqrt(h - 1) = 1.
FALLBACK_SIGMA = 0.15

MAX_VIEWERS = 1_000_000_000.0
CENT = Decimal("0.01")


def parse_time(text):
    """Parse an ISO-8601 timestamp into an aware UTC datetime.

    Accepts a trailing ``Z``, explicit offsets, fractional seconds and
    naive values (taken as UTC).  Returns None for anything unusable.
    """
    if not isinstance(text, str) or not text.strip():
        return None
    cleaned = text.strip()
    if cleaned.endswith(("Z", "z")):
        cleaned = cleaned[:-1] + "+00:00"
    try:
        moment = datetime.fromisoformat(cleaned)
    except ValueError:
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc)


def hour_one_viewers(case):
    """The case's hour-1 viewer print, at least 1 (0 if unknown).

    Rows explicitly labelled hour 1 win; only when no row carries an
    hour label at all is the first plausible row accepted.
    """
    known = case.get("known")
    rows = known if isinstance(known, list) else [known]
    rows = [row for row in rows if isinstance(row, dict)]
    for wanted in (1, None):
        for row in rows:
            if row.get("hour") != wanted:
                continue
            try:
                viewers = float(row.get("viewers"))
            except (TypeError, ValueError):
                continue
            if math.isfinite(viewers):
                return max(viewers, 1.0)
    return 0.0


def release_day_series(comparable):
    """Map hour index (1..24) -> mean viewers on the comparable's day.

    Points are bucketed by their rounded hour offset from the
    comparable's own day start; several points in one hour are
    averaged.  Returns an empty dict when the day cannot be placed.
    """
    start = parse_time(comparable.get("day_start"))
    if start is None:
        return {}
    buckets = {}
    for point in comparable.get("hourly") or []:
        if not isinstance(point, dict):
            continue
        when = parse_time(point.get("from") or point.get("at"))
        if when is None:
            continue
        try:
            viewers = float(point.get("viewers"))
        except (TypeError, ValueError):
            continue
        if not math.isfinite(viewers):
            continue
        # Round half up (not Python's banker's rounding): a point
        # exactly on a half-hour boundary belongs to the later hour.
        offset = (when - start).total_seconds() / 3600.0
        hour = math.floor(offset + 0.5) + 1
        if 1 <= hour <= 24:
            buckets.setdefault(hour, []).append(viewers)
    return {hour: sum(vals) / len(vals) for hour, vals in buckets.items()}


def usable_comparables(case):
    """Release-day series worth learning from, as (hour1, series)."""
    kept = []
    for comparable in case.get("comparables") or []:
        if not isinstance(comparable, dict):
            continue
        series = release_day_series(comparable)
        first = series.get(1)
        if first is None or first <= 0:
            continue  # no launch moment to anchor ratios on
        if len(series) < MIN_DAY_HOURS:
            continue  # too fragmentary to describe a launch day
        kept.append((first, series))
    return kept


def kernel_log_weight(first_viewers, anchor):
    """Log of the Gaussian weight for a comparable.

    Kept in log space so callers can shift the whole set before
    exponentiating: a case whose hour-1 audience is far from every
    comparable would otherwise underflow every weight to 0.0.
    """
    distance = (math.log(first_viewers) - math.log(anchor)) / KERNEL_BANDWIDTH
    return -0.5 * distance * distance


def kernel_weights(firsts, anchor):
    """Gaussian weights, rescaled so the largest is exactly 1.

    Weighted quantiles only depend on relative weights, so the
    rescaling changes nothing mathematically; it only keeps the
    arithmetic well away from floating-point underflow.
    """
    log_weights = [kernel_log_weight(first, anchor) for first in firsts]
    peak = max(log_weights)
    return [math.exp(value - peak) for value in log_weights]


def weighted_quantile(sorted_pairs, tau):
    """Weighted quantile of (value, weight) pairs sorted by value.

    Each point is placed at the midpoint of its own weight slab,
    (cumulative weight - half its weight) / total weight, and the
    level ``tau`` is linearly interpolated between neighbouring
    points, clamped to the end values outside their range.
    """
    total = sum(weight for _, weight in sorted_pairs)
    if total <= 0:
        return sorted_pairs[len(sorted_pairs) // 2][0]
    positions = []
    cumulative = 0.0
    for _, weight in sorted_pairs:
        cumulative += weight
        positions.append((cumulative - 0.5 * weight) / total)
    if tau <= positions[0]:
        return sorted_pairs[0][0]
    if tau >= positions[-1]:
        return sorted_pairs[-1][0]
    for idx in range(1, len(positions)):
        if tau <= positions[idx]:
            low_pos, high_pos = positions[idx - 1], positions[idx]
            low_val = sorted_pairs[idx - 1][0]
            high_val = sorted_pairs[idx][0]
            if high_pos <= low_pos:
                return high_val
            share = (tau - low_pos) / (high_pos - low_pos)
            return low_val + share * (high_val - low_val)
    return sorted_pairs[-1][0]


def fallback_log_quantiles(hour, levels):
    """Persistence log-ratios: flat median, spread ~ sqrt(hours)."""
    spread = FALLBACK_SIGMA * math.sqrt(hour - 1)
    return [z * spread for z in Z_SCORES[: len(levels)]]


def raw_log_quantiles(case, anchor, levels):
    """Unsmoothed per-hour log-ratio quantiles, hours 2..24.

    Returns None when there are too few usable comparables, so the
    caller uses the pure persistence fallback for every hour.
    """
    kept = usable_comparables(case)
    if len(kept) < MIN_COMPARABLES:
        return None
    weights = kernel_weights([first for first, _ in kept], anchor)
    curves = {}
    for hour in range(2, 25):
        pairs = []
        for (first, series), weight in zip(kept, weights):
            viewers = series.get(hour)
            if viewers is None or viewers <= 0:
                continue
            pairs.append((math.log(viewers / first), weight))
        if not pairs:
            # No comparable covers this hour: borrow the fallback so
            # the curve stays complete and smoothable.
            curves[hour] = fallback_log_quantiles(hour, levels)
            continue
        pairs.sort(key=lambda pair: pair[0])
        curves[hour] = [weighted_quantile(pairs, tau) for tau in levels]
    return curves


def smooth_curves(curves, levels):
    """Replace each hour by the mean of itself and its neighbours."""
    smoothed = {}
    for hour in range(2, 25):
        window = [curves[h] for h in (hour - 1, hour, hour + 1) if h in curves]
        smoothed[hour] = [
            sum(row[col] for row in window) / len(window)
            for col in range(len(levels))
        ]
    return smoothed


def round_half_up(value):
    """Round to 2 decimals, halves away from zero, as a float."""
    return float(Decimal(repr(value)).quantize(CENT, rounding=ROUND_HALF_UP))


def finalise_row(anchor, log_quantiles):
    """Turn one hour's log-ratios into ordered, in-range viewers."""
    row = []
    for log_ratio in log_quantiles:
        value = round_half_up(anchor * math.exp(log_ratio))
        value = min(max(value, 0.01), MAX_VIEWERS)
        if row and value < row[-1]:
            value = row[-1]  # quantiles never decrease within an hour
        row.append(value)
    return row


def forecast_case(case):
    """Build the {"hours": [...]} forecast object for one case."""
    levels = case.get("levels") or DEFAULT_LEVELS
    levels = [float(level) for level in levels]
    anchor = hour_one_viewers(case)
    if anchor <= 0:
        anchor = 1.0
    curves = raw_log_quantiles(case, anchor, levels)
    if curves is None:
        curves = {
            hour: fallback_log_quantiles(hour, levels) for hour in range(2, 25)
        }
    else:
        curves = smooth_curves(curves, levels)
    hours = [
        {"hour": hour, "q": finalise_row(anchor, curves[hour])}
        for hour in range(2, 25)
    ]
    return {"hours": hours}


def main():
    input_dir = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
    output_dir = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(input_dir.glob("*.in")):
        with path.open("r", encoding="utf-8") as handle:
            case = json.load(handle)
        result = forecast_case(case)
        case_id = str(case.get("id") or path.stem)
        out_path = output_dir / f"{case_id}.out"
        payload = json.dumps(result, separators=(",", ":"))
        out_path.write_text(payload, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
