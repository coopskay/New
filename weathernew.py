import argparse
import io
import json
import re
import sys
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import eccodes
import numpy as np
import requests
from scipy.optimize import minimize
from scipy.stats import norm, skewnorm

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")


# ─── Logging ──────────────────────────────────────────────────────────────────

class _Tee:
    """Write to both stdout and an appending log file, with run-start timestamps."""

    def __init__(self, log_path: str):
        self._stdout = sys.stdout
        self._log    = open(log_path, "a", encoding="utf-8")
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%Mz")
        self._log.write(f"\n{'='*70}\n  Run started {stamp}\n{'='*70}\n")
        self._log.flush()

    def write(self, data: str):
        self._stdout.write(data)
        self._log.write(data)

    def flush(self):
        self._stdout.flush()
        self._log.flush()

    def close(self):
        self._log.close()
        sys.stdout = self._stdout


# ─── Constants ────────────────────────────────────────────────────────────────

TEMP_UNIT  = "fahrenheit"
ENS_MODEL  = "ecmwf_ifs025"            # 51-member ECMWF IFS ensemble
AIFS_MODEL = "ecmwf_aifs025_ensemble"  # ECMWF AIFS ensemble

# (lat, lon, display_label, station_id)
PRESETS: dict[str, tuple] = {
    "kord": (41.9742, -87.9073,  "Chicago O'Hare (KORD)",     "KORD"),
    "ksea": (47.4499, -122.3118, "Seattle-Tacoma (KSEA)",     "KSEA"),
    "kdal": (32.8481, -96.8512,  "Dallas Love Field (KDAL)",  "KDAL"),
    "klga": (40.7772, -73.8726,  "New York LaGuardia (KLGA)", "KLGA"),
    "katl": (33.6407, -84.4277,  "Atlanta (KATL)",             "KATL"),
    "kaus": (30.1975, -97.6664,  "Austin-Bergstrom (KAUS)",   "KAUS"),
    "khou": (29.6454, -95.2789,  "Houston Hobby (KHOU)",      "KHOU"),
    "kmia": (25.7959, -80.2870,  "Miami (KMIA)",               "KMIA"),
}

TIMEZONES: dict[str, str] = {
    "KORD": "America/Chicago",
    "KSEA": "America/Los_Angeles",
    "KDAL": "America/Chicago",
    "KLGA": "America/New_York",
    "KATL": "America/New_York",
    "KAUS": "America/Chicago",
    "KHOU": "America/Chicago",
    "KMIA": "America/New_York",
}

# WFO codes for NWS Area Forecast Discussion
WFO_MAP: dict[str, str] = {
    "KORD": "LOT",
    "KSEA": "SEW",
    "KDAL": "FWD",
    "KLGA": "OKX",
    "KATL": "FFC",
    "KAUS": "EWX",
    "KHOU": "HGX",
    "KMIA": "MFL",
}

# Short city names for Polymarket slug generation
CITY_SLUG_NAMES: dict[str, str] = {
    "KORD": "chicago",
    "KSEA": "seattle",
    "KDAL": "dallas",
    "KLGA": "nyc",
    "KATL": "atlanta",
    "KAUS": "austin",
    "KHOU": "houston",
    "KMIA": "miami",
}


# ─── Utilities ────────────────────────────────────────────────────────────────

def _k_to_f(k):
    return (k - 273.15) * 9 / 5 + 32 if k is not None else None

def _c_to_f(c):
    return c * 9 / 5 + 32 if c is not None else None

def fetch_json(url, extra_headers=None):
    headers = {"User-Agent": "weathernew.py/2.0"}
    if extra_headers:
        headers.update(extra_headers)
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=20) as resp:
        return json.loads(resp.read())

def fetch_json_params(base, params):
    return fetch_json(f"{base}?{urllib.parse.urlencode(params)}")

def _polymarket_slug(station_id: str, target_date: str) -> str | None:
    """Generate expected Polymarket slug: highest-temperature-in-{city}-on-{date}."""
    city = CITY_SLUG_NAMES.get(station_id)
    if not city:
        return None
    dt        = datetime.strptime(target_date, "%Y-%m-%d")
    date_slug = f"{dt.strftime('%B').lower()}-{dt.day}-{dt.year}"
    return f"highest-temperature-in-{city}-on-{date_slug}"


# ─── NWS Hourly Forecast ──────────────────────────────────────────────────────

def _fetch_nws(lat, lon, target_date, now_utc) -> dict:
    """Returns {'output': list[str], 'max_temp_f': float|None}."""
    lines: list[str] = []
    result: dict     = {"max_temp_f": None}
    try:
        points_data   = fetch_json(f"https://api.weather.gov/points/{lat},{lon}")
        hourly_url    = points_data["properties"]["forecastHourly"]
        forecast_data = fetch_json(hourly_url)
        periods       = forecast_data["properties"]["periods"]
        raw_upd       = forecast_data["properties"].get("updateTime", "")
        upd           = raw_upd[:16].replace("T", " ") + "z" if raw_upd else "unknown"

        max_temp, max_unit = None, None
        for p in periods:
            if p["startTime"][:10] != target_date:
                continue
            start_utc = datetime.fromisoformat(p["startTime"]).astimezone(timezone.utc)
            if start_utc < now_utc:
                continue
            if max_temp is None or p["temperature"] > max_temp:
                max_temp = p["temperature"]
                max_unit = p["temperatureUnit"]

        if max_temp is None:
            lines.append(f"[NWS]        no data  (updated {upd})")
        else:
            lines.append(f"[NWS]        {max_temp}°{max_unit}  (updated {upd})")
            result["max_temp_f"] = float(max_temp) if max_unit == "F" else _c_to_f(float(max_temp))
    except Exception as e:
        lines.append(f"[NWS]        error ({e})")

    result["output"] = lines
    return result


# ─── Open-Meteo point-forecast (shared core) ─────────────────────────────────

def _fetch_openmeteo_point(
    lat, lon, target_date, now_utc, tz_name,
    model: str, tag: str,
    api_base: str = "https://api.open-meteo.com/v1/forecast",
    with_wind_dewp: bool = True,
) -> dict:
    """
    Fetch hourly T2m (+ optionally dewpoint + wind) from Open-Meteo for one model.
    Returns dict with keys: output, temp_f, dewp_f, wind_mph, peak_time, hourly_temps_f.
    'tag' is the bracketed label prefix used in output lines, e.g. '[HRRR]      '.
    """
    lines: list[str] = []
    result: dict = {"temp_f": None, "dewp_f": None, "wind_mph": None,
                    "peak_time": None, "hourly_temps_f": []}
    try:
        tz       = ZoneInfo(tz_name) if tz_name else timezone.utc
        tz_param = tz_name or "UTC"
        hvars    = "temperature_2m,dewpoint_2m,windspeed_10m" if with_wind_dewp else "temperature_2m"
        params   = {
            "latitude": lat, "longitude": lon,
            "hourly": hvars,
            "models": model,
            "temperature_unit": TEMP_UNIT,
            "windspeed_unit": "mph",
            "timezone": tz_param,
            "start_date": target_date, "end_date": target_date,
        }
        data   = fetch_json_params(api_base, params)
        hourly = data.get("hourly", {})
        cutoff = now_utc.astimezone(tz).strftime("%Y-%m-%dT%H:%M")
        times  = hourly.get("time", [])
        temps  = hourly.get("temperature_2m", [])
        dewps  = hourly.get("dewpoint_2m", [None] * len(times))
        winds  = hourly.get("windspeed_10m", [None] * len(times))

        valid = [(t, v, d, w)
                 for t, v, d, w in zip(times, temps, dewps, winds)
                 if v is not None and t >= cutoff]

        if not valid:
            lines.append(f"{tag} no data for remainder of day")
        else:
            max_time, max_temp, max_dewp, max_wind = max(valid, key=lambda x: x[1])
            lo         = min(v for _, v, _, _ in valid)
            peak_label = "local" if tz_name else "z"
            extras     = []
            if max_dewp is not None:
                extras.append(f"dewp {max_dewp:.1f}°F")
            if max_wind is not None:
                extras.append(f"wind {max_wind:.1f}mph")
            ext_str = ("  " + "  ".join(extras)) if extras else ""
            lines.append(f"{tag} {max_temp:.1f}°F  "
                         f"(peak {max_time[-5:]} {peak_label}, range {lo:.1f}–{max_temp:.1f}°F"
                         f"{ext_str})")
            result["temp_f"]         = max_temp
            result["dewp_f"]         = max_dewp
            result["wind_mph"]       = max_wind
            result["peak_time"]      = max_time
            result["hourly_temps_f"] = [(t, v) for t, v, _, _ in valid]
    except Exception as e:
        lines.append(f"{tag} error ({e})")

    result["output"] = lines
    return result


def _fetch_hrrr(lat, lon, target_date, now_utc, tz_name) -> dict:
    return _fetch_openmeteo_point(lat, lon, target_date, now_utc, tz_name,
                                  model="ncep_hrrr_conus", tag="[HRRR]      ")

def _fetch_nbm(lat, lon, target_date, now_utc, tz_name) -> dict:
    return _fetch_openmeteo_point(lat, lon, target_date, now_utc, tz_name,
                                  model="ncep_nbm_conus", tag="[NBM]       ")

def _fetch_gfs(lat, lon, target_date, now_utc, tz_name) -> dict:
    return _fetch_openmeteo_point(lat, lon, target_date, now_utc, tz_name,
                                  model="gfs_seamless", tag="[GFS]       ")

def _fetch_ifs_det(lat, lon, target_date, now_utc, tz_name) -> dict:
    return _fetch_openmeteo_point(lat, lon, target_date, now_utc, tz_name,
                                  model="ecmwf_ifs025", tag="[ECMWF IFS] ",
                                  api_base="https://api.open-meteo.com/v1/ecmwf",
                                  with_wind_dewp=False)

def _fetch_aifs_det(lat, lon, target_date, now_utc, tz_name) -> dict:
    return _fetch_openmeteo_point(lat, lon, target_date, now_utc, tz_name,
                                  model="ecmwf_aifs025", tag="[AIFS det]  ",
                                  with_wind_dewp=False)


# ─── NBM QMD percentile forecast (GRIB2 from NOMADS) ─────────────────────────

_QMD_BASE    = "https://nomads.ncep.noaa.gov/pub/data/nccf/com/blend/v5.0"
_QMD_CYCLES  = [0, 6, 12, 18]
_QMD_REGION  = "co"
_QMD_PCTS    = [10, 25, 50, 75, 90]
_QMD_WORKERS = 10
_QMD_TIMEOUT = 60
_qmd_grid_cache: dict[tuple, int] = {}


def _qmd_urls(date: datetime, cycle: int, fxx: int) -> tuple[str, str]:
    stem = (f"{_QMD_BASE}/blend.{date.strftime('%Y%m%d')}/{cycle:02d}/qmd"
            f"/blend.t{cycle:02d}z.qmd.f{fxx:03d}.{_QMD_REGION}.grib2")
    return stem, stem + ".idx"


def _qmd_latest_cycle() -> tuple[datetime, int]:
    now = datetime.now(timezone.utc).replace(tzinfo=None, minute=0, second=0, microsecond=0)
    for dh in range(0, 48):
        t = now - timedelta(hours=dh)
        for c in sorted(_QMD_CYCLES, reverse=True):
            if t.hour >= c:
                cand = t.replace(hour=c, minute=0, second=0, microsecond=0)
                _, idx_url = _qmd_urls(cand, c, 1)
                try:
                    if requests.head(idx_url, timeout=_QMD_TIMEOUT).status_code == 200:
                        return cand, c
                except Exception:
                    pass
                break
    raise RuntimeError("No recent QMD run found on NOMADS")


def _qmd_fetch_idx(date: datetime, cycle: int, fxx: int) -> list[str] | None:
    _, url = _qmd_urls(date, cycle, fxx)
    try:
        r = requests.get(url, timeout=_QMD_TIMEOUT)
        return r.text.splitlines() if r.status_code == 200 else None
    except Exception:
        return None


def _qmd_parse_tmp_recs(idx_lines: list[str]) -> dict[int, dict]:
    """Return {pct: {start, end}} for desired TMP 2m percentile levels."""
    recs = {}
    for i, line in enumerate(idx_lines):
        if ":TMP:2 m above ground:" not in line:
            continue
        for p in _QMD_PCTS:
            if line.endswith(f":{p}% level"):
                start = int(line.split(":")[1])
                end   = int(idx_lines[i + 1].split(":")[1]) - 1 if i + 1 < len(idx_lines) else -1
                recs[p] = {"start": start, "end": end}
                break
    return recs


def _qmd_fetch_bytes(grib_url: str, rec: dict) -> bytes | None:
    hdr = f"bytes={rec['start']}-" if rec["end"] == -1 else f"bytes={rec['start']}-{rec['end']}"
    try:
        r = requests.get(grib_url, headers={"Range": hdr}, timeout=_QMD_TIMEOUT)
        return r.content if r.status_code in (200, 206) else None
    except Exception:
        return None


def _qmd_decode(grib_bytes: bytes, lat: float, lon: float) -> float | None:
    """Decode one GRIB2 message and return the value at (lat, lon). NOT thread-safe."""
    key        = (round(lat, 4), round(lon, 4))
    target_lon = lon % 360
    try:
        msg = eccodes.codes_new_from_message(grib_bytes)
        if msg is None:
            return None
        try:
            if key not in _qmd_grid_cache:
                lats = eccodes.codes_get_array(msg, "latitudes")
                lons = eccodes.codes_get_array(msg, "longitudes")
                _qmd_grid_cache[key] = int(np.argmin((lats - lat) ** 2 + (lons - target_lon) ** 2))
            vals = eccodes.codes_get_array(msg, "values")
        finally:
            eccodes.codes_release(msg)
        return float(vals[_qmd_grid_cache[key]])
    except Exception:
        return None


def _nbm_daily_max_pctls(vals_by_fxx_pct: dict, fxx_range: list) -> dict[int, float] | None:
    """
    Compute daily-max temperature percentiles using a spread-adjusted normal.

    Combines the hour-of-peak spread (sigma_hour, from P10/P90 at the peak hour)
    with timing uncertainty (sigma_timing, from the spread of P50 values across
    the top hours) into sigma_total = sqrt(sigma_hour^2 + sigma_timing^2).

    All internal values are in Kelvin; returned values are in Fahrenheit.
    """
    hourly_p50_k = {fxx: vals_by_fxx_pct[(fxx, 50)]
                    for fxx in fxx_range if (fxx, 50) in vals_by_fxx_pct}
    if not hourly_p50_k:
        return None

    peak_fxx   = max(hourly_p50_k, key=lambda f: hourly_p50_k[f])
    peak_p50_k = hourly_p50_k[peak_fxx]

    # sigma_hour: intensity spread at the peak hour
    p10_k = vals_by_fxx_pct.get((peak_fxx, 10))
    p90_k = vals_by_fxx_pct.get((peak_fxx, 90))
    p25_k = vals_by_fxx_pct.get((peak_fxx, 25))
    p75_k = vals_by_fxx_pct.get((peak_fxx, 75))
    if p10_k is not None and p90_k is not None:
        sigma_hour_k = (p90_k - p10_k) / 2.564
    elif p25_k is not None and p75_k is not None:
        sigma_hour_k = (p75_k - p25_k) / 1.349
    else:
        sigma_hour_k = 1.0  # fallback ~1.8°F

    # sigma_timing: how much the daily max could vary due to timing uncertainty.
    # Use the spread of the top-3 hourly P50 values as a proxy.
    sorted_p50 = sorted(hourly_p50_k.values(), reverse=True)
    top        = sorted_p50[:min(3, len(sorted_p50))]
    sigma_timing_k = (top[0] - top[-1]) / 2.0 if len(top) > 1 else 0.0

    sigma_total_k = (sigma_hour_k ** 2 + sigma_timing_k ** 2) ** 0.5
    sigma_total_f = sigma_total_k * 9 / 5
    peak_p50_f    = _k_to_f(peak_p50_k)

    # Normal-distribution quantiles
    z = {10: -1.282, 25: -0.674, 50: 0.0, 75: 0.674, 90: 1.282}
    return {p: peak_p50_f + z[p] * sigma_total_f for p in _QMD_PCTS}


def nbm_qmd_high_percentiles(
    lat: float, lon: float, now_utc: datetime,
    tz_name: str, target_date: str, obs_high: float | None = None,
) -> dict | None:
    """
    Fetch NBM QMD TMP percentiles for the remaining hours of target_date.
    Returns dict with keys: output, peak_hour, daily_max, run_str, cycle, or None on failure.
      peak_hour  – percentiles at the single hour with highest P50
      daily_max  – spread-adjusted distribution (sigma_hour + sigma_timing combined)
    """
    lines: list[str] = []
    try:
        date, cycle = _qmd_latest_cycle()
    except RuntimeError as e:
        return {"output": [f"[NBM QMD]    {e}"], "peak_hour": None, "daily_max": None}

    cycle_age_hours = (now_utc - date.replace(tzinfo=timezone.utc)).total_seconds() / 3600
    if cycle_age_hours > 8:
        lines.append(f"[NBM QMD]    [warn] stale cycle: {cycle_age_hours:.1f}h old "
                     f"(run {date.strftime('%Y-%m-%d')} {cycle:02d}z)")

    tz  = ZoneInfo(tz_name)
    y, m, d = int(target_date[:4]), int(target_date[5:7]), int(target_date[8:10])
    sod_local = datetime(y, m, d,  0,  0,  0, tzinfo=tz)
    eod_local = datetime(y, m, d, 23, 59, 59, tzinfo=tz)
    from_local = max(now_utc.astimezone(tz), sod_local)
    init_utc   = date.replace(hour=cycle, tzinfo=timezone.utc)

    fxx_range = [
        fxx for fxx in range(1, 113)
        if from_local <= (init_utc + timedelta(hours=fxx)).astimezone(tz) <= eod_local
    ]
    if not fxx_range:
        return {"output": ["[NBM QMD]    no remaining forecast hours today"],
                "peak_hour": None, "daily_max": None}

    # Fetch idx files in parallel
    idx_by_fxx: dict[int, list[str] | None] = {}
    with ThreadPoolExecutor(max_workers=_QMD_WORKERS) as ex:
        futs = {ex.submit(_qmd_fetch_idx, date, cycle, fxx): fxx for fxx in fxx_range}
        for fut in as_completed(futs):
            idx_by_fxx[futs[fut]] = fut.result()

    # Build download tasks
    tasks: list[tuple[int, int, str, dict]] = []
    for fxx in fxx_range:
        lines_idx = idx_by_fxx.get(fxx)
        if not lines_idx:
            continue
        grib_url, _ = _qmd_urls(date, cycle, fxx)
        for pct, rec in _qmd_parse_tmp_recs(lines_idx).items():
            tasks.append((fxx, pct, grib_url, rec))

    # Fetch GRIB bytes in parallel
    raw_by_key: dict[tuple[int, int], bytes | None] = {}
    with ThreadPoolExecutor(max_workers=_QMD_WORKERS) as ex:
        futs = {ex.submit(_qmd_fetch_bytes, grib_url, rec): (fxx, pct)
                for fxx, pct, grib_url, rec in tasks}
        for fut in as_completed(futs):
            raw_by_key[futs[fut]] = fut.result()

    # Decode sequentially (eccodes not thread-safe)
    vals_by_fxx_pct: dict[tuple[int, int], float] = {}
    for (fxx, pct), raw in raw_by_key.items():
        if raw is None:
            continue
        v = _qmd_decode(raw, lat, lon)
        if v is not None:
            vals_by_fxx_pct[(fxx, pct)] = v

    # Peak-hour percentiles (existing logic: pick hour with highest P50)
    peak_p50_fxx = None
    peak_p50_val = -999.0
    for fxx in fxx_range:
        if (fxx, 50) in vals_by_fxx_pct and vals_by_fxx_pct[(fxx, 50)] > peak_p50_val:
            peak_p50_val = vals_by_fxx_pct[(fxx, 50)]
            peak_p50_fxx = fxx

    peak_hour: dict[int, float] = {}
    if peak_p50_fxx is not None:
        for p in _QMD_PCTS:
            if (peak_p50_fxx, p) in vals_by_fxx_pct:
                peak_hour[p] = _k_to_f(vals_by_fxx_pct[(peak_p50_fxx, p)])

    if len(peak_hour) < 3:
        return {"output": ["[NBM QMD]    insufficient data"],
                "peak_hour": None, "daily_max": None}

    # Daily-max percentiles (spread-adjusted)
    daily_max = _nbm_daily_max_pctls(vals_by_fxx_pct, fxx_range)

    run_str = f"{date.strftime('%Y-%m-%d')} {cycle:02d}z"
    first_valid = (init_utc + timedelta(hours=fxx_range[0])).astimezone(ZoneInfo(tz_name))
    last_valid  = (init_utc + timedelta(hours=fxx_range[-1])).astimezone(ZoneInfo(tz_name))
    window_str  = (f"{first_valid.strftime('%I%p').lstrip('0').lower()}–"
                   f"{last_valid.strftime('%I%p').lstrip('0').lower()}")

    peak_p50_hour_local = (init_utc + timedelta(hours=peak_p50_fxx)).astimezone(ZoneInfo(tz_name))
    peak_time_str = peak_p50_hour_local.strftime("%I%p").lstrip("0").lower()

    ph_str = "  ".join(f"{p}th:{peak_hour[p]:.1f}" for p in _QMD_PCTS if p in peak_hour)
    lines.append(f"[NBM QMD]    peak-hour → {ph_str}°F")
    lines.append(f"             (peak {peak_time_str} local, run {run_str}, window {window_str})")
    lines.append(f"             [intensity-only uncertainty at single hour]")

    if daily_max:
        dm_str = "  ".join(f"{p}th:{daily_max[p]:.1f}" for p in _QMD_PCTS if p in daily_max)
        lines.append(f"[NBM QMD]    daily-max → {dm_str}°F")
        lines.append(f"             [spread-adjusted: sigma_hour + sigma_timing combined]")

    if obs_high is not None:
        floored = [p for p in _QMD_PCTS if p in peak_hour and peak_hour[p] < obs_high]
        if len(floored) >= 3:
            lines.append(f"  [note] obs_high {obs_high:.1f}°F floors QMD P{max(floored)} and below")
        elif obs_high > peak_hour.get(50, 0):
            lines.append(f"  [note] obs_high {obs_high:.1f}°F above QMD P50 ({peak_hour.get(50):.1f}°F)")

    return {"output": lines, "peak_hour": peak_hour, "daily_max": daily_max, "run_str": run_str}


# ─── Ensemble statistics helper ───────────────────────────────────────────────

def _ens_stats_lines(member_highs, weights=None) -> list[str]:
    """Return formatted ensemble statistics as a list of strings (no printing)."""
    lines: list[str] = []
    maxes = np.array(member_highs)

    if weights is not None:
        w        = np.array(weights)
        w        = w / w.sum()
        mean_val = float(np.average(maxes, weights=w))
        std_val  = float(np.sqrt(np.average((maxes - mean_val) ** 2, weights=w)))
        p        = _weighted_percentile(maxes, w, [5, 10, 25, 50, 75, 90, 95])
    else:
        mean_val = float(np.mean(maxes))
        std_val  = float(np.std(maxes))
        p        = np.percentile(maxes, [5, 10, 25, 50, 75, 90, 95])

    pct_labels = ["5th", "10th", "25th", "50th", "75th", "90th", "95th"]
    pct_str    = "  ".join(f"{l}:{v:.1f}" for l, v in zip(pct_labels, p))
    lines.append(f"  Mean {mean_val:.1f}  Median {p[3]:.1f}  Std {std_val:.1f}"
                 f"  Range {np.min(maxes):.1f}–{np.max(maxes):.1f}°F")
    lines.append(f"  {pct_str}°F")

    bar_max = 24
    if weights is not None:
        wcounts, edges = np.histogram(maxes, bins=10, weights=w)
        lines.append(f"\n  Distribution ({len(member_highs)} members, weighted):")
        peak = wcounts.max()
        for i in range(len(wcounts)):
            lo, hi = edges[i], edges[i + 1]
            bar = "█" * int(wcounts[i] / peak * bar_max) if peak > 0 else ""
            lines.append(f"  {lo:>5.1f}–{hi:<5.1f}°F  {wcounts[i]*100:>4.1f}%  {bar}")
    else:
        counts, edges = np.histogram(maxes, bins=10)
        lines.append(f"\n  Distribution ({len(member_highs)} members):")
        for i in range(len(counts)):
            lo, hi = edges[i], edges[i + 1]
            bar    = "█" * int(counts[i] / counts.max() * bar_max)
            lines.append(f"  {lo:>5.1f}–{hi:<5.1f}°F  {counts[i]:>3}  {bar}")

    p5_t  = round(float(p[0]))
    p95_t = round(float(p[6]))
    thresh = list(range(p5_t, p95_t + 1, 5))
    probs  = []
    for t in thresh:
        if weights is not None:
            prob = float(np.sum(w[maxes >= t])) * 100
        else:
            prob = float(np.mean(maxes >= t)) * 100
        probs.append(f"P(>={t}) {prob:.0f}%")
    if probs:
        lines.append("\n  " + "  ".join(probs))

    return lines


def _weighted_percentile(values, weights, percentiles):
    idx         = np.argsort(values)
    sorted_vals = values[idx]
    cumw        = np.cumsum(weights[idx])
    cumw       /= cumw[-1]
    return np.interp([p / 100 for p in percentiles], cumw, sorted_vals)


def _print_ens_stats(member_highs, weights=None):
    for line in _ens_stats_lines(member_highs, weights):
        print(line)


# ─── ECMWF IFS Ensemble ───────────────────────────────────────────────────────

def _fetch_ifs_ens(lat, lon, target_date, now_utc, tz_name) -> dict:
    """Returns {'output': list[str], 'member_highs': list[float], 'model_run': str}."""
    lines: list[str] = []
    result: dict = {"member_highs": [], "model_run": "unknown"}
    try:
        tz       = ZoneInfo(tz_name) if tz_name else timezone.utc
        tz_param = tz_name or "UTC"
        data     = fetch_json_params("https://ensemble-api.open-meteo.com/v1/ensemble", {
            "latitude": lat, "longitude": lon,
            "hourly": "temperature_2m",
            "models": ENS_MODEL,
            "start_date": target_date, "end_date": target_date,
            "temperature_unit": TEMP_UNIT,
            "timezone": tz_param,
        })
        hourly   = data.get("hourly", {})
        times    = hourly.get("time", [])
        model_run = times[0].replace("T", " ") if times else "unknown"
        cutoff   = now_utc.astimezone(tz).strftime("%Y-%m-%dT%H:%M")

        member_keys = sorted(k for k in hourly if k.startswith("temperature_2m_member"))
        if "temperature_2m" in hourly and "temperature_2m" not in member_keys:
            member_keys = ["temperature_2m"] + member_keys

        member_highs = []
        for key in member_keys:
            vals = [v for t, v in zip(times, hourly.get(key, []))
                    if v is not None and t >= cutoff]
            if vals:
                member_highs.append(max(vals))

        result["member_highs"] = member_highs
        result["model_run"]    = model_run
        lines.append(f"\n[ECMWF ENS]  {len(member_highs)} members, model run {model_run}")
        if not member_highs:
            lines.append("  No data returned")
        else:
            lines.extend(_ens_stats_lines(member_highs))
    except Exception as e:
        lines.append(f"\n[ECMWF ENS]  error ({e})")

    result["output"] = lines
    return result


# ─── ECMWF AIFS Ensemble ──────────────────────────────────────────────────────

def _fetch_aifs_ens(lat, lon, target_date, now_utc, tz_name) -> dict:
    """Returns {'output': list[str], 'member_highs': list[float], 'model_run': str}."""
    lines: list[str] = []
    result: dict = {"member_highs": [], "model_run": "unknown"}
    try:
        tz       = ZoneInfo(tz_name) if tz_name else timezone.utc
        tz_param = tz_name or "UTC"
        data     = fetch_json_params("https://ensemble-api.open-meteo.com/v1/ensemble", {
            "latitude": lat, "longitude": lon,
            "hourly": "temperature_2m",
            "models": AIFS_MODEL,
            "start_date": target_date, "end_date": target_date,
            "temperature_unit": TEMP_UNIT,
            "timezone": tz_param,
        })
        hourly    = data.get("hourly", {})
        times     = hourly.get("time", [])
        model_run = times[0].replace("T", " ") if times else "unknown"
        cutoff    = now_utc.astimezone(tz).strftime("%Y-%m-%dT%H:%M")

        member_keys = sorted(k for k in hourly if k.startswith("temperature_2m_member"))
        if "temperature_2m" in hourly and "temperature_2m" not in member_keys:
            member_keys = ["temperature_2m"] + member_keys

        member_highs = []
        for key in member_keys:
            vals = [v for t, v in zip(times, hourly.get(key, []))
                    if v is not None and t >= cutoff]
            if vals:
                member_highs.append(max(vals))

        result["member_highs"] = member_highs
        result["model_run"]    = model_run
        lines.append(f"\n[AIFS ENS]   {len(member_highs)} members, model run {model_run}")
        if not member_highs:
            lines.append("  No data returned")
        else:
            lines.extend(_ens_stats_lines(member_highs))
    except Exception as e:
        lines.append(f"\n[AIFS ENS]   unavailable ({e})")

    result["output"] = lines
    return result


# ─── ECMWF IFS Ensemble post-processed ───────────────────────────────────────

def _fetch_ens_pp(
    lat, lon, label, target_date,
    hrrr_val, nbm_val, gfs_val, ifs_val, aifs_val,
    now_utc, obs_high, tz_name,
) -> dict:
    """
    Bias-correct + spread-inflate the IFS ensemble using a freshness-weighted
    anchor across HRRR, NBM, GFS, IFS-det, and AIFS-det.
    Returns {'output': list[str], 'member_highs': list[float], 'anchor': float|None}.
    """
    lines: list[str] = []
    result: dict = {"member_highs": [], "anchor": None}

    anchor_candidates = [
        ("HRRR", hrrr_val,  0.7, 0.0),
        ("NBM",  nbm_val,   1.0, 0.0),
        ("GFS",  gfs_val,   0.5, 2.0),
        ("IFS",  ifs_val,   0.4, 9.0),   # age estimated below
        ("AIFS", aifs_val,  0.4, 9.0),
    ]
    live = [(name, val, wb, age)
            for name, val, wb, age in anchor_candidates if val is not None]
    if not live:
        lines.append(f"\n[ECMWF ENS PP]  {label} — skipped (no anchor available)")
        result["output"] = lines
        return result

    try:
        tz       = ZoneInfo(tz_name) if tz_name else timezone.utc
        tz_param = tz_name or "UTC"
        data     = fetch_json_params("https://ensemble-api.open-meteo.com/v1/ensemble", {
            "latitude": lat, "longitude": lon,
            "hourly": "temperature_2m",
            "models": ENS_MODEL,
            "start_date": target_date, "end_date": target_date,
            "temperature_unit": TEMP_UNIT,
            "timezone": tz_param,
        })
    except Exception as e:
        lines.append(f"\n[ECMWF ENS PP]  {label} — unavailable ({e})")
        result["output"] = lines
        return result

    hourly = data.get("hourly", {})
    times  = hourly.get("time", [])
    cutoff = now_utc.astimezone(tz).strftime("%Y-%m-%dT%H:%M") if now_utc else ""

    # Approximate IFS/AIFS age from first timestamp
    ifs_age_approx = 9.0
    if times and now_utc is not None:
        try:
            run_dt       = datetime.fromisoformat(times[0]).astimezone(timezone.utc)
            ifs_age_approx = max(0.0, (now_utc - run_dt).total_seconds() / 3600)
        except Exception:
            pass

    # Rebuild with actual IFS age
    anchor_candidates_adj = [
        ("HRRR", hrrr_val,  0.7, 0.0),
        ("NBM",  nbm_val,   1.0, 0.0),
        ("GFS",  gfs_val,   0.5, 2.0),
        ("IFS",  ifs_val,   0.4, ifs_age_approx),
        ("AIFS", aifs_val,  0.4, ifs_age_approx),
    ]
    anchor_weights = {
        name: wb * float(np.exp(-age / 18))
        for name, val, wb, age in anchor_candidates_adj if val is not None
    }
    anchor_vals = {
        name: val
        for name, val, _, _ in anchor_candidates_adj if val is not None
    }
    total_w = sum(anchor_weights.values())
    anchor  = sum(anchor_weights[k] * anchor_vals[k] for k in anchor_weights) / total_w

    member_keys = sorted(k for k in hourly if k.startswith("temperature_2m_member"))
    if "temperature_2m" in hourly and "temperature_2m" not in member_keys:
        member_keys = ["temperature_2m"] + member_keys

    raw_highs = []
    for key in member_keys:
        vals = [v for t, v in zip(times, hourly.get(key, []))
                if v is not None and t >= cutoff]
        if vals:
            raw_highs.append(max(vals))

    if not raw_highs:
        lines.append(f"\n[ECMWF ENS PP]  {label} — no data returned")
        result["output"] = lines
        return result
    if len(raw_highs) < 50:
        lines.append(f"  [warn] only {len(raw_highs)} members (expected ≥50)")

    raw       = np.array(raw_highs)
    ens_mean  = float(np.mean(raw))
    delta     = anchor - ens_mean

    sigma_ens     = float(np.std(raw))
    anchor_pts    = list(anchor_vals.values())
    sigma_anchor  = max(float(np.std(anchor_pts)) if len(anchor_pts) >= 2 else 0.0, 0.5)
    sigma_ens_corr = sigma_ens * 1.15
    sigma_target  = float(np.sqrt(sigma_ens_corr ** 2 + sigma_anchor ** 2))
    scale         = sigma_target / sigma_ens if sigma_ens > 0 else 1.0

    SCALE_HYBRID_THRESHOLD = 1.5
    shifted = raw + delta
    if scale <= SCALE_HYBRID_THRESHOLD:
        inflated = anchor + (shifted - anchor) * scale
    else:
        capped_scale  = SCALE_HYBRID_THRESHOLD
        inflated      = anchor + (shifted - anchor) * capped_scale
        sigma_residual = float(np.sqrt(max(sigma_target ** 2 - (sigma_ens * capped_scale) ** 2, 0.0)))
        seed           = hash(f"{label}_{target_date}") & 0xFFFFFFFF
        rng            = np.random.default_rng(seed)
        inflated       = inflated + rng.normal(0.0, sigma_residual, size=inflated.shape)
        lines.append(f"  [note] scale={scale:.2f}x > {SCALE_HYBRID_THRESHOLD} — hybrid: "
                     f"cap={capped_scale}x + σ_extra={sigma_residual:.2f}°F")

    if obs_high is not None:
        raw_pts = [v for v in anchor_vals.values()]
        if raw_pts and all(v < obs_high for v in raw_pts):
            lines.append(f"  [note] all model forecasts below obs_high ({obs_high:.1f}°F); "
                         f"daily high effectively locked at obs_high")
        inflated = np.maximum(inflated, obs_high)

    w_str = "  ".join(f"{k}:{anchor_weights[k]:.2f}" for k in anchor_weights)
    lines.append(f"\n[ECMWF ENS PP]  {label} — {target_date}  "
                 f"({len(raw_highs)} members, anchor={anchor:.1f}°F [{w_str}], "
                 f"delta={delta:+.1f}°F, σ_ens={sigma_ens:.2f}°F, "
                 f"σ_anchor={sigma_anchor:.2f}°F, scale={scale:.2f}x)")
    lines.extend(_ens_stats_lines(list(inflated)))

    result["member_highs"] = list(inflated)
    result["anchor"]       = anchor
    result["output"]       = lines
    return result


# ─── METAR / Obs ──────────────────────────────────────────────────────────────

def _metar_obs_utc(ob):
    """Return observation time as UTC-aware datetime."""
    epoch = ob.get("obsTime")
    if epoch is not None:
        return datetime.fromtimestamp(int(epoch), tz=timezone.utc)
    ts = ob.get("reportTime", "")
    if ts:
        return datetime.fromisoformat(ts).astimezone(timezone.utc)
    return None


def _fetch_current_metar(station_id) -> dict:
    """Returns {'output': list[str], 'temp_f': float|None, 'time_str': str|None}."""
    result: dict = {"temp_f": None, "time_str": None, "output": []}

    # NWS 5-minute ASOS (preferred)
    try:
        data     = fetch_json(f"https://api.weather.gov/stations/{station_id}/observations?limit=5")
        features = data.get("features", [])
        for f in features:
            props  = f.get("properties", {})
            t_obj  = props.get("temperature", {}) or {}
            temp_c = t_obj.get("value")
            if temp_c is None:
                continue
            ts      = props.get("timestamp", "")
            obs_utc = datetime.fromisoformat(ts).astimezone(timezone.utc)
            temp_f  = temp_c * 9 / 5 + 32
            result["temp_f"]   = temp_f
            result["time_str"] = obs_utc.strftime("%H:%Mz")
            result["output"]   = [f"[OBS CURRENT] {temp_f:.1f}°F  current temperature  "
                                   f"(updated {result['time_str']})"]
            return result
    except Exception as e:
        result["output"].append(f"  [NWS obs fetch failed: {e}]")

    # Fallback: aviationweather.gov
    try:
        data = fetch_json(
            f"https://aviationweather.gov/api/data/metar?ids={station_id}&format=json")
        if data:
            ob      = data[0]
            temp_c  = ob.get("temp")
            obs_utc = _metar_obs_utc(ob)
            if temp_c is not None and obs_utc is not None:
                temp_f             = temp_c * 9 / 5 + 32
                result["temp_f"]   = temp_f
                result["time_str"] = obs_utc.strftime("%H:%Mz")
                result["output"]   = [f"[OBS CURRENT] {temp_f:.1f}°F  current temperature  "
                                       f"(updated {result['time_str']})"]
                return result
    except Exception as e:
        result["output"].append(f"  [METAR fetch failed: {e}]")

    result["output"].append("[OBS CURRENT] no data")
    return result


def _fetch_obs_history(station_id, target_date, tz_name) -> dict:
    """
    Fetch all METAR/SPECI obs for target_date from aviationweather.gov.
    Returns dict with keys: output, high_all_f, high_routine_f, high_all_local,
    high_all_utc, high_routine_local, high_routine_utc, history (list of dicts).
    """
    result: dict = {
        "high_all_f": None, "high_routine_f": None,
        "high_all_local": None, "high_all_utc": None,
        "high_routine_local": None, "high_routine_utc": None,
        "history": [], "output": [],
    }
    try:
        data = fetch_json(
            f"https://aviationweather.gov/api/data/metar?ids={station_id}&format=json&hours=24")
        if not data:
            result["output"] = ["[OBS HIGH]    no data"]
            return result

        tz      = ZoneInfo(tz_name) if tz_name else timezone.utc
        history = []
        best_all     = None
        best_routine = None

        for ob in data:
            temp_c  = ob.get("temp")
            obs_utc = _metar_obs_utc(ob)
            if temp_c is None or obs_utc is None:
                continue
            obs_local = obs_utc.astimezone(tz)
            if obs_local.strftime("%Y-%m-%d") != target_date:
                continue

            temp_f    = temp_c * 9 / 5 + 32
            dewp_c    = ob.get("dewp")
            dewp_f    = _c_to_f(dewp_c) if dewp_c is not None else None
            obs_type  = ob.get("metarType", "METAR")
            is_routine = obs_type != "SPECI"

            history.append({
                "time_utc":   obs_utc.strftime("%H:%Mz"),
                "time_local": obs_local.strftime("%H:%M %Z"),
                "temp_f":     round(temp_f, 1),
                "dewp_f":     round(dewp_f, 1) if dewp_f is not None else None,
                "type":       "ROUTINE" if is_routine else "SPECI",
            })

            if best_all is None or temp_f > best_all["temp_f"]:
                best_all = {
                    "temp_f": temp_f,
                    "local":  obs_local.strftime("%H:%M %Z"),
                    "utc":    obs_utc.strftime("%H:%Mz"),
                }
            if is_routine and (best_routine is None or temp_f > best_routine["temp_f"]):
                best_routine = {
                    "temp_f": temp_f,
                    "local":  obs_local.strftime("%H:%M %Z"),
                    "utc":    obs_utc.strftime("%H:%Mz"),
                }

        # Sort history chronologically
        history.sort(key=lambda x: x["time_utc"])
        result["history"] = history

        lines: list[str] = []
        if best_all:
            result["high_all_f"]     = best_all["temp_f"]
            result["high_all_local"] = best_all["local"]
            result["high_all_utc"]   = best_all["utc"]
            lines.append(f"[OBS HIGH]    {best_all['temp_f']:.1f}°F  "
                         f"observed high (all obs)  "
                         f"({best_all['local']} / {best_all['utc']})")
        else:
            lines.append("[OBS HIGH]    no data")

        if best_routine:
            result["high_routine_f"]     = best_routine["temp_f"]
            result["high_routine_local"] = best_routine["local"]
            result["high_routine_utc"]   = best_routine["utc"]
            suffix = "  ← ENS floor" if best_routine["temp_f"] != (best_all or {}).get("temp_f") else ""
            lines.append(f"[OBS ROUTINE] {best_routine['temp_f']:.1f}°F  "
                         f"routine METARs only  "
                         f"({best_routine['local']} / {best_routine['utc']}){suffix}")
        else:
            lines.append("[OBS ROUTINE] no data")

        # Observation history table
        if history:
            lines.append("\n[OBS HISTORY] (chronological, today)")
            lines.append(f"  {'UTC':>5}  {'Local':>10}  {'Temp':>6}  {'Dewp':>6}  {'Type'}")
            lines.append(f"  {'─'*5}  {'─'*10}  {'─'*6}  {'─'*6}  {'─'*7}")
            for h in history:
                dewp_s = f"{h['dewp_f']:>5.1f}" if h['dewp_f'] is not None else "   n/a"
                flag   = "" if h["type"] == "ROUTINE" else " ★SPECI"
                lines.append(f"  {h['time_utc']:>5}  {h['time_local']:>10}  "
                              f"{h['temp_f']:>5.1f}°F  {dewp_s}°F  {h['type']}{flag}")

        result["output"] = lines
    except Exception as e:
        result["output"] = [f"[OBS HIGH]    error ({e})"]

    return result


# ─── NWS Area Forecast Discussion ─────────────────────────────────────────────

def _fetch_nws_afd(station_id) -> dict:
    """
    Fetch the NWS Area Forecast Discussion for the station's WFO.
    Returns {'output': list[str], 'short_term': str|None, 'wfo': str|None}.
    """
    result: dict = {"short_term": None, "wfo": None, "output": []}
    wfo = WFO_MAP.get(station_id)
    if not wfo:
        return result
    result["wfo"] = wfo
    try:
        data  = fetch_json(f"https://api.weather.gov/products?type=AFD&location={wfo}")
        items = data.get("@graph", [])
        if not items:
            result["output"] = [f"[NWS AFD]    no AFD found for WFO {wfo}"]
            return result

        product_id = items[0].get("id")
        if not product_id:
            result["output"] = [f"[NWS AFD]    product ID missing for WFO {wfo}"]
            return result

        product = fetch_json(f"https://api.weather.gov/products/{product_id}")
        text    = product.get("productText", "")

        # Try SHORT TERM → NEAR TERM → SYNOPSIS
        excerpt = None
        for section in ("SHORT TERM", "NEAR TERM", "SYNOPSIS"):
            pat   = rf"\.{section}[^\n]*\.\n(.*?)(?=\n\.[A-Z]|\Z)"
            match = re.search(pat, text, re.DOTALL | re.IGNORECASE)
            if match:
                excerpt = match.group(1).strip()[:2000]
                break

        if not excerpt:
            excerpt = text[:1000]

        result["short_term"] = excerpt
        issuance = items[0].get("issuanceTime", "")[:16].replace("T", " ") + "z"
        lines    = [f"\n[NWS AFD]    WFO {wfo}  (issued {issuance})",
                    "─" * 60]
        lines   += [f"  {ln}" for ln in excerpt.splitlines()]
        lines.append("─" * 60)
        result["output"] = lines
    except Exception as e:
        result["output"] = [f"[NWS AFD]    error ({e})"]

    return result


# ─── Polymarket ───────────────────────────────────────────────────────────────

def _fetch_polymarket(slug, label) -> dict:
    """Returns {'output': list[str], 'rows': list[(label, prob)]|None, 'slug': str}."""
    result: dict = {"rows": None, "slug": slug, "output": []}
    if not slug:
        result["output"] = ["\n[POLYMARKET]  no slug available"]
        return result
    try:
        data = fetch_json(f"https://gamma-api.polymarket.com/events?slug={slug}")
    except Exception as e:
        result["output"] = [f"\n[POLYMARKET]  unavailable ({e})"]
        return result

    if not data:
        result["output"] = [f"\n[POLYMARKET]  no event found for slug: {slug}"]
        return result

    event   = data[0] if isinstance(data, list) else data
    markets = event.get("markets", [])
    if not markets:
        result["output"] = ["\n[POLYMARKET]  no markets found"]
        return result

    rows = []
    for m in markets:
        question = m.get("question", "")
        prices   = m.get("outcomePrices", "[]")
        if isinstance(prices, str):
            try:
                prices = json.loads(prices)
            except json.JSONDecodeError:
                continue
        try:
            yes_prob = float(prices[0]) * 100
        except (IndexError, ValueError, TypeError):
            continue
        match      = re.search(r"be (.+?)\s+on\s+\w+", question, re.IGNORECASE)
        range_label = re.sub(r"^between\s+", "", match.group(1) if match else question)
        rows.append((range_label, yes_prob))

    fetched_at = datetime.now(timezone.utc).strftime("%H:%Mz")
    lines      = [f"\n[POLYMARKET]  {label}  (fetched {fetched_at})"]
    bar_max    = 20
    peak       = max((p for _, p in rows), default=1)
    for range_label, prob in rows:
        bar = "█" * int(prob / peak * bar_max) if peak > 0 else ""
        lines.append(f"  {range_label:<22}  {prob:>5.1f}%  {bar}")

    result["rows"]   = rows
    result["output"] = lines
    return result


# ─── Distribution fitting + bucket probabilities ──────────────────────────────

def _fit_nbm_distribution(nbm_pctls: dict[int, float]):
    """
    Fit skew-normal to five NBM percentile values.
    Returns a frozen scipy distribution, or None on failure.
    """
    probs  = [p / 100 for p in _QMD_PCTS]
    target = np.array([nbm_pctls[p] for p in _QMD_PCTS])
    med    = nbm_pctls[50]
    iqr    = nbm_pctls[75] - nbm_pctls[25]
    scale0 = max(iqr / 1.349, 0.5)

    def sse(params):
        a, loc, sc = params
        if sc <= 0:
            return 1e10
        return float(np.sum((skewnorm.ppf(probs, a, loc, sc) - target) ** 2))

    try:
        res = minimize(sse, x0=[0.0, med, scale0], method="Nelder-Mead",
                       options={"xatol": 0.01, "fatol": 0.01, "maxiter": 5000})
        a, loc, sc = res.x
        if sc > 0 and abs(a) < 50 and res.fun < 1.0:
            return skewnorm(a, loc, sc)
    except Exception:
        pass

    sigma = (nbm_pctls[90] - nbm_pctls[10]) / 2.564
    return norm(med, max(sigma, 0.1))


def _nbm_bucket_probs(nbm_pctls: dict[int, float],
                      market_rows: list,
                      obs_high: float | None = None) -> list[float] | None:
    """
    Fit distribution to NBM percentiles, apply obs_high floor, integrate over
    each market bucket. Returns list of probabilities (0–100) or None on failure.
    """
    if nbm_pctls is None or not market_rows:
        return None
    dist = _fit_nbm_distribution(nbm_pctls)
    if dist is None:
        return None

    floor   = obs_high if obs_high is not None else -np.inf
    p_above = 1.0 - dist.cdf(floor)
    if p_above <= 0:
        return None

    probs = []
    for label, _ in market_rows:
        m_below = re.match(r"(\d+\.?\d*)°?F?\s+or\s+below",  label, re.IGNORECASE)
        m_range = re.match(r"(\d+\.?\d*)\s*[-–]\s*(\d+\.?\d*)°?F?", label, re.IGNORECASE)
        m_above = re.match(r"(\d+\.?\d*)°?F?\s+or\s+higher", label, re.IGNORECASE)

        if m_below:
            hi  = float(m_below.group(1)) + 0.5
            raw = dist.cdf(hi) - dist.cdf(floor)
        elif m_range:
            lo  = max(float(m_range.group(1)) - 0.5, floor)
            hi  = float(m_range.group(2)) + 0.5
            raw = dist.cdf(hi) - dist.cdf(lo)
        elif m_above:
            lo  = max(float(m_above.group(1)) - 0.5, floor)
            raw = 1.0 - dist.cdf(lo)
        else:
            probs.append(float("nan"))
            continue

        probs.append(max(0.0, raw / p_above) * 100)
    return probs


def _ens_bucket_probs(member_highs: list[float],
                      market_rows: list,
                      obs_high: float | None = None) -> list[float] | None:
    """Compute bucket probabilities directly from ensemble member daily highs."""
    if not member_highs or not market_rows:
        return None
    members = np.array(member_highs)
    if obs_high is not None:
        members = np.maximum(members, obs_high)

    probs = []
    for label, _ in market_rows:
        m_below = re.match(r"(\d+\.?\d*)°?F?\s+or\s+below",  label, re.IGNORECASE)
        m_range = re.match(r"(\d+\.?\d*)\s*[-–]\s*(\d+\.?\d*)°?F?", label, re.IGNORECASE)
        m_above = re.match(r"(\d+\.?\d*)°?F?\s+or\s+higher", label, re.IGNORECASE)

        if m_below:
            hi   = float(m_below.group(1)) + 0.5
            prob = float(np.mean(members < hi)) * 100
        elif m_range:
            lo   = float(m_range.group(1)) - 0.5
            hi   = float(m_range.group(2)) + 0.5
            prob = float(np.mean((members >= lo) & (members < hi))) * 100
        elif m_above:
            lo   = float(m_above.group(1)) - 0.5
            prob = float(np.mean(members >= lo)) * 100
        else:
            probs.append(float("nan"))
            continue
        probs.append(prob)
    return probs


# ─── Percentile comparison ────────────────────────────────────────────────────

def _print_pctl_compare(ens_pp_members, nbm_pctls, aifs_members=None) -> None:
    if ens_pp_members is None or nbm_pctls is None:
        return

    pcts     = [10, 25, 50, 75, 90]
    ens_vals = np.percentile(np.array(ens_pp_members), pcts)
    nbm_vals = [nbm_pctls.get(p) for p in pcts]
    if any(v is None for v in nbm_vals):
        return

    col = 7
    hdr = "             " + "".join(f"  {str(p)+'th':>{col}}" for p in pcts)
    print(f"\n[PCTL COMPARE]")
    print(hdr)
    print("  ENS PP    " + "".join(f"  {v:>{col}.1f}" for v in ens_vals))
    print("  NBM QMD   " + "".join(f"  {v:>{col}.1f}" for v in nbm_vals))
    if aifs_members:
        aifs_vals = np.percentile(np.array(aifs_members), pcts)
        print("  AIFS ENS  " + "".join(f"  {v:>{col}.1f}" for v in aifs_vals))
        delta_aifs = [float(e) - float(a) for e, a in zip(ens_vals, aifs_vals)]
        print("  Δ ENS-AIFS" + "".join(f"  {d:>+{col}.1f}" for d in delta_aifs))

    deltas = [float(e) - float(n) for e, n in zip(ens_vals, nbm_vals)]
    print("  Δ ENS-NBM " + "".join(f"  {d:>+{col}.1f}" for d in deltas))

    tail_warn = abs(deltas[0]) > 1.5 or abs(deltas[-1]) > 1.5
    if tail_warn:
        print("  [WARN] ENS-NBM percentile disagreement > 1.5°F at tails")


# ─── Edge comparison table ────────────────────────────────────────────────────

def compare_ensemble_to_market(
    ens_pp_members, market_rows,
    nbm_probs: list[float] | None = None,
    aifs_probs: list[float] | None = None,
) -> tuple[list[str], list[dict]]:
    """
    Side-by-side bucket probability table: ENS PP, NBM, AIFS vs Market.
    Returns (output_lines, edge_data_list).
    Consensus flag (★) when ≥2 of 3 sources agree in direction with |edge| ≥ 5pp.
    """
    lines: list[str] = []
    edges: list[dict] = []

    if market_rows is None or ens_pp_members is None or len(ens_pp_members) == 0:
        return lines, edges

    members  = np.array(ens_pp_members)
    has_nbm  = nbm_probs  is not None and len(nbm_probs)  == len(market_rows)
    has_aifs = aifs_probs is not None and len(aifs_probs) == len(market_rows)

    # Header
    hdr_cols = f"{'Bucket':<24}  {'ENS PP':>8}"
    sep_cols = f"  {'─'*22}  {'─'*8}"
    if has_nbm:
        hdr_cols += f"  {'NBM QMD':>8}"
        sep_cols += f"  {'─'*8}"
    if has_aifs:
        hdr_cols += f"  {'AIFS ENS':>8}"
        sep_cols += f"  {'─'*8}"
    hdr_cols += f"  {'Market':>8}"
    sep_cols += f"  {'─'*8}"
    if has_nbm or has_aifs:
        hdr_cols += f"  {'Edge(ENS)':>10}"
        sep_cols += f"  {'─'*10}"
    if has_nbm:
        hdr_cols += f"  {'Edge(NBM)':>10}"
        sep_cols += f"  {'─'*10}"
    if has_aifs:
        hdr_cols += f"  {'Edge(AIFS)':>10}"
        sep_cols += f"  {'─'*10}"
    lines.append(f"\n{hdr_cols}")
    lines.append(sep_cols)

    agreed: list[tuple[str, float, float | None, float | None]] = []

    for i, (lbl, mkt_prob) in enumerate(market_rows):
        m_below = re.match(r"(\d+\.?\d*)°?F?\s+or\s+below",  lbl, re.IGNORECASE)
        m_range = re.match(r"(\d+\.?\d*)\s*[-–]\s*(\d+\.?\d*)°?F?", lbl, re.IGNORECASE)
        m_above = re.match(r"(\d+\.?\d*)°?F?\s+or\s+higher", lbl, re.IGNORECASE)

        if m_below:
            hi       = float(m_below.group(1))
            ens_prob = float(np.mean(members < hi + 0.5)) * 100
        elif m_range:
            lo, hi   = float(m_range.group(1)), float(m_range.group(2))
            ens_prob = float(np.mean((members >= lo - 0.5) & (members < hi + 0.5))) * 100
        elif m_above:
            lo       = float(m_above.group(1))
            ens_prob = float(np.mean(members >= lo - 0.5)) * 100
        else:
            ens_prob = float("nan")

        nbm_p  = nbm_probs[i]  if has_nbm  else float("nan")
        aifs_p = aifs_probs[i] if has_aifs else float("nan")

        ens_edge  = ens_prob  - mkt_prob if not np.isnan(ens_prob)  else float("nan")
        nbm_edge  = nbm_p     - mkt_prob if not np.isnan(nbm_p)     else float("nan")
        aifs_edge = aifs_p    - mkt_prob if not np.isnan(aifs_p)    else float("nan")

        # Build row string
        row = f"  {lbl:<22}  {ens_prob:>7.1f}%"
        if has_nbm:
            row += f"  {nbm_p:>7.1f}%" if not np.isnan(nbm_p) else f"  {'n/a':>8}"
        if has_aifs:
            row += f"  {aifs_p:>7.1f}%" if not np.isnan(aifs_p) else f"  {'n/a':>8}"
        row += f"  {mkt_prob:>7.1f}%"

        flag = ""
        # Consensus: ≥2 of 3 sources agree in direction with |edge| ≥ 5pp
        valid_edges = [(e, src) for e, src in [(ens_edge, "ENS"), (nbm_edge, "NBM"), (aifs_edge, "AIFS")]
                       if not np.isnan(e) and abs(e) >= 5]
        if len(valid_edges) >= 2:
            directions = set(np.sign(e) for e, _ in valid_edges)
            if len(directions) == 1:
                flag = "  ★"
                agreed.append((lbl, ens_edge, nbm_edge if has_nbm else None,
                                aifs_edge if has_aifs else None))

        if not np.isnan(ens_edge):
            row += f"  {ens_edge:>+9.1f}pp"
        if has_nbm:
            row += f"  {nbm_edge:>+9.1f}pp" if not np.isnan(nbm_edge) else f"  {'n/a':>10}"
        if has_aifs:
            row += f"  {aifs_edge:>+9.1f}pp" if not np.isnan(aifs_edge) else f"  {'n/a':>10}"

        lines.append(row + flag)
        edges.append({
            "bucket":    lbl,
            "ens_prob":  round(ens_prob,  1) if not np.isnan(ens_prob)  else None,
            "nbm_prob":  round(nbm_p,     1) if not np.isnan(nbm_p)     else None,
            "aifs_prob": round(aifs_p,    1) if not np.isnan(aifs_p)    else None,
            "market_prob": mkt_prob,
            "ens_edge":  round(ens_edge,  1) if not np.isnan(ens_edge)  else None,
            "nbm_edge":  round(nbm_edge,  1) if not np.isnan(nbm_edge)  else None,
            "aifs_edge": round(aifs_edge, 1) if not np.isnan(aifs_edge) else None,
        })

    if agreed:
        lines.append(f"\n  ★ Consensus (≥2 sources agree, |edge| ≥ 5pp, same direction):")
        for lbl, e_ens, e_nbm, e_aifs in agreed:
            direction = "OVER" if e_ens > 0 else "UNDER"
            parts = [f"ENS {e_ens:+.1f}pp"]
            if e_nbm  is not None and not np.isnan(e_nbm):
                parts.append(f"NBM {e_nbm:+.1f}pp")
            if e_aifs is not None and not np.isnan(e_aifs):
                parts.append(f"AIFS {e_aifs:+.1f}pp")
            lines.append(f"    {lbl}: {direction}  " + "  ".join(parts))

    return lines, edges


# ─── AI-optimised summary block ───────────────────────────────────────────────

def _print_ai_summary(
    label, station_id, target_date, now_utc, tz_name,
    obs_r: dict, metar_r: dict,
    nws_r: dict, hrrr_r: dict, nbm_r: dict, gfs_r: dict,
    ifs_det_r: dict, aifs_det_r: dict,
    nbm_qmd_r: dict | None,
    ifs_ens_r: dict, aifs_ens_r: dict, ens_pp_r: dict,
    poly_r: dict, edge_data: list[dict],
    afd_r: dict,
) -> None:
    sep = "═" * 70
    print(f"\n{sep}")
    print(f"  [AI SUMMARY]  {label} — {target_date}")
    print(sep)

    # ── Observation floor ──
    obs_high_f = obs_r.get("high_routine_f")
    tz         = ZoneInfo(tz_name) if tz_name else timezone.utc
    eod        = datetime(int(target_date[:4]), int(target_date[5:7]),
                          int(target_date[8:10]), 23, 59, 59, tzinfo=tz)
    hours_left = max(0.0, (eod.astimezone(timezone.utc) - now_utc).total_seconds() / 3600)

    obs_floor_str = (f"{obs_high_f:.1f}°F  ({obs_r.get('high_routine_local')})"
                     if obs_high_f else "n/a")
    cur_temp_f = metar_r.get("temp_f")
    cur_temp_str = (f"{cur_temp_f:.1f}°F  ({metar_r.get('time_str')})"
                    if cur_temp_f else "n/a")
    print(f"\nOBS FLOOR (routine METAR high so far):  {obs_floor_str}")
    print(f"CURRENT TEMP:  {cur_temp_str}")
    print(f"HOURS REMAINING IN DAY:  {hours_left:.1f}h")

    # ── Deterministic model consensus ──
    det_vals = {
        "HRRR":  hrrr_r.get("temp_f"),
        "NBM":   nbm_r.get("temp_f"),
        "GFS":   gfs_r.get("temp_f"),
        "IFS":   ifs_det_r.get("temp_f"),
        "AIFS":  aifs_det_r.get("temp_f"),
        "NWS":   nws_r.get("max_temp_f"),
    }
    live_vals = {k: v for k, v in det_vals.items() if v is not None}
    print(f"\nDETERMINISTIC MODEL CONSENSUS")
    vals_str = "  ".join(f"{k}={v:.1f}°F" for k, v in live_vals.items())
    print(f"  Models:    {vals_str if vals_str else 'none available'}")
    if live_vals:
        arr        = np.array(list(live_vals.values()))
        consensus  = float(np.mean(arr))
        spread_std = float(np.std(arr))
        agree_flag = "AGREE" if spread_std < 2.0 else "DISAGREE"
        print(f"  Consensus: {consensus:.1f}°F  (σ={spread_std:.1f}°F across {len(arr)} models — models {agree_flag})")
        if obs_high_f:
            gap = consensus - obs_high_f
            if gap > 0:
                print(f"  Obs is {gap:.1f}°F below consensus — some heating still expected")
            else:
                print(f"  Obs has already met/exceeded consensus — daily high likely locked")
    else:
        consensus = None

    # ── Probabilistic distributions ──
    print(f"\nPROBABILISTIC DISTRIBUTIONS (remaining daily max)")
    p_labels = [10, 25, 50, 75, 90]

    if nbm_qmd_r and nbm_qmd_r.get("peak_hour"):
        ph = nbm_qmd_r["peak_hour"]
        ph_str = "  ".join(f"P{p}={ph[p]:.1f}" for p in p_labels if p in ph)
        print(f"  NBM QMD peak-hour:  {ph_str}°F")
    if nbm_qmd_r and nbm_qmd_r.get("daily_max"):
        dm = nbm_qmd_r["daily_max"]
        dm_str = "  ".join(f"P{p}={dm[p]:.1f}" for p in p_labels if p in dm)
        print(f"  NBM QMD daily-max:  {dm_str}°F")

    if ens_pp_r.get("member_highs"):
        pp  = np.percentile(np.array(ens_pp_r["member_highs"]), p_labels)
        pp_str = "  ".join(f"P{p}={v:.1f}" for p, v in zip(p_labels, pp))
        print(f"  ENS PP (IFS-based): {pp_str}°F")
    if aifs_ens_r.get("member_highs"):
        ap  = np.percentile(np.array(aifs_ens_r["member_highs"]), p_labels)
        ap_str = "  ".join(f"P{p}={v:.1f}" for p, v in zip(p_labels, ap))
        print(f"  AIFS ENS (raw):     {ap_str}°F")

    # Dew point context at peak hour
    for tag, r in [("HRRR", hrrr_r), ("NBM", nbm_r), ("GFS", gfs_r)]:
        if r.get("dewp_f") is not None:
            print(f"  Dew point at {tag} peak: {r['dewp_f']:.1f}°F  "
                  f"(spread {r['temp_f']:.1f}–{r['dewp_f']:.1f} = "
                  f"{r['temp_f'] - r['dewp_f']:.0f}°F dep)")
            break  # show only one

    # ── Market odds + edge table ──
    if poly_r.get("rows"):
        print(f"\nMARKET ODDS + EDGE TABLE  (slug: {poly_r.get('slug', 'n/a')})")
        if edge_data:
            has_nbm  = any(e.get("nbm_edge")  is not None for e in edge_data)
            has_aifs = any(e.get("aifs_edge") is not None for e in edge_data)
            header = f"  {'Bucket':<22}  {'Market':>7}  {'ENS PP':>7}"
            if has_nbm:
                header += f"  {'NBM':>7}"
            if has_aifs:
                header += f"  {'AIFS':>7}"
            header += f"  {'ENS edge':>9}"
            if has_nbm:
                header += f"  {'NBM edge':>9}"
            if has_aifs:
                header += f"  {'AIFSedge':>9}"
            print(header)
            for e in edge_data:
                ep = e.get('ens_prob')
                row = (f"  {e['bucket']:<22}  {e['market_prob']:>6.1f}%"
                       f"  {ep:>6.1f}%" if ep is not None else
                       f"  {e['bucket']:<22}  {e['market_prob']:>6.1f}%  {'n/a':>7}")
                if has_nbm:
                    np_ = e.get('nbm_prob')
                    row += f"  {np_:>6.1f}%" if np_ is not None else f"  {'n/a':>7}"
                if has_aifs:
                    ap = e.get('aifs_prob')
                    row += f"  {ap:>6.1f}%" if ap is not None else f"  {'n/a':>7}"
                ens_e = e.get('ens_edge')
                row += f"  {ens_e:>+8.1f}pp" if ens_e is not None else f"  {'n/a':>9}"
                if has_nbm:
                    ne = e.get('nbm_edge')
                    row += f"  {ne:>+8.1f}pp" if ne is not None else f"  {'n/a':>9}"
                if has_aifs:
                    ae = e.get('aifs_edge')
                    row += f"  {ae:>+8.1f}pp" if ae is not None else f"  {'n/a':>9}"
                # Consensus flag
                cons_edges = [x for x in [ens_e, e.get('nbm_edge'), e.get('aifs_edge')]
                              if x is not None and abs(x) >= 5]
                if len(cons_edges) >= 2 and len(set(int(x > 0) for x in cons_edges)) == 1:
                    row += "  ★"
                print(row)

    # ── Consensus edge flags ──
    consensus_buckets = [
        e for e in edge_data
        if sum(1 for x in [e.get('ens_edge'), e.get('nbm_edge'), e.get('aifs_edge')]
               if x is not None and abs(x) >= 5) >= 2
        and len(set(int(x > 0) for x in [e.get('ens_edge'), e.get('nbm_edge'), e.get('aifs_edge')]
                    if x is not None and abs(x) >= 5)) == 1
    ]
    if consensus_buckets:
        print(f"\nCONSENSUS EDGE FLAGS (≥2 of 3 model sources agree, |edge| ≥ 5pp):")
        for e in consensus_buckets:
            direction = "OVER" if (e.get('ens_edge') or 0) > 0 else "UNDER"
            parts = []
            for src, k in [("ENS", "ens_edge"), ("NBM", "nbm_edge"), ("AIFS", "aifs_edge")]:
                v = e.get(k)
                if v is not None:
                    parts.append(f"{src} {v:+.1f}pp")
            print(f"  ★ {e['bucket']}: {direction}  " + "  ".join(parts))
    else:
        print(f"\nCONSENSUS EDGE FLAGS: none flagged (no bucket with ≥2 sources at ≥5pp)")

    # ── Qualitative risk flags ──
    print(f"\nQUALITATIVE RISK FLAGS:")
    flags_printed = 0

    if live_vals:
        if spread_std > 3.0:
            print(f"  ⚠ High inter-model spread (σ={spread_std:.1f}°F) — significant model uncertainty")
            flags_printed += 1
        elif spread_std < 1.0:
            print(f"  ✓ Models in tight agreement (σ={spread_std:.1f}°F)")
            flags_printed += 1

    # AIFS vs IFS disagreement
    if ifs_det_r.get("temp_f") and aifs_det_r.get("temp_f"):
        diff = abs(ifs_det_r["temp_f"] - aifs_det_r["temp_f"])
        if diff > 3.0:
            print(f"  ⚠ AIFS ({aifs_det_r['temp_f']:.1f}°F) vs IFS ({ifs_det_r['temp_f']:.1f}°F) "
                  f"disagree by {diff:.1f}°F — high model uncertainty")
            flags_printed += 1

    # Obs floor proximity
    if obs_high_f and consensus:
        gap = consensus - obs_high_f
        if gap < 2.0:
            print(f"  ⚠ Obs floor ({obs_high_f:.1f}°F) within 2°F of consensus — daily high nearly locked")
            flags_printed += 1
        elif gap > 8.0:
            print(f"  ⚠ Large gap between obs ({obs_high_f:.1f}°F) and consensus ({consensus:.1f}°F) — "
                  f"forecast highly dependent on remaining heating")
            flags_printed += 1

    # Hours remaining
    if hours_left < 3:
        print(f"  ⚠ Only {hours_left:.1f}h remaining in day — limited time for surprises")
        flags_printed += 1

    # Dew point flag
    for tag, r in [("HRRR", hrrr_r), ("NBM", nbm_r), ("GFS", gfs_r)]:
        if r.get("temp_f") and r.get("dewp_f"):
            dep = r["temp_f"] - r["dewp_f"]
            if dep < 10:
                print(f"  ⚠ Low dew point depression at {tag} peak ({dep:.0f}°F) — "
                      f"high moisture may suppress max temp")
                flags_printed += 1
            break

    if flags_printed == 0:
        print(f"  ✓ No significant risk flags identified")

    # ── NWS AFD excerpt ──
    if afd_r.get("short_term"):
        print(f"\nNWS AFD SHORT TERM (WFO {afd_r.get('wfo', '?')}):")
        for ln in afd_r["short_term"][:800].splitlines():
            print(f"  {ln}")

    print(f"\n{sep}\n")


# ─── JSON output ──────────────────────────────────────────────────────────────

def _build_json_payload(
    label, station_id, lat, lon, target_date, now_utc, tz_name,
    obs_r, metar_r, nws_r, hrrr_r, nbm_r, gfs_r,
    ifs_det_r, aifs_det_r, nbm_qmd_r,
    ifs_ens_r, aifs_ens_r, ens_pp_r,
    poly_r, edge_data, afd_r,
) -> dict:
    def _pctls(members):
        if not members:
            return None
        arr = np.array(members)
        return {str(p): round(float(np.percentile(arr, p)), 2)
                for p in [5, 10, 25, 50, 75, 90, 95]}

    return {
        "run_utc":     now_utc.isoformat(),
        "target_date": target_date,
        "station": {"id": station_id, "label": label,
                    "lat": lat, "lon": lon, "timezone": tz_name},
        "observations": {
            "current_temp_f":     metar_r.get("temp_f"),
            "current_temp_time":  metar_r.get("time_str"),
            "daily_high_all_f":   obs_r.get("high_all_f"),
            "daily_high_routine_f": obs_r.get("high_routine_f"),
            "history":            obs_r.get("history", []),
        },
        "models": {
            "nws":      {"max_temp_f": nws_r.get("max_temp_f")},
            "hrrr":     {k: hrrr_r.get(k) for k in ("temp_f", "dewp_f", "wind_mph", "peak_time")},
            "nbm":      {k: nbm_r.get(k)  for k in ("temp_f", "dewp_f", "wind_mph", "peak_time")},
            "gfs":      {k: gfs_r.get(k)  for k in ("temp_f", "dewp_f", "wind_mph", "peak_time")},
            "ifs_det":  {"max_temp_f": ifs_det_r.get("temp_f")},
            "aifs_det": {"max_temp_f": aifs_det_r.get("temp_f")},
            "nbm_qmd":  {
                "peak_hour":  nbm_qmd_r.get("peak_hour") if nbm_qmd_r else None,
                "daily_max":  nbm_qmd_r.get("daily_max") if nbm_qmd_r else None,
                "run_str":    nbm_qmd_r.get("run_str")   if nbm_qmd_r else None,
            },
            "ifs_ens":     {"member_highs": ifs_ens_r.get("member_highs"),
                            "percentiles": _pctls(ifs_ens_r.get("member_highs"))},
            "ifs_ens_pp":  {"member_highs": ens_pp_r.get("member_highs"),
                            "anchor": ens_pp_r.get("anchor"),
                            "percentiles": _pctls(ens_pp_r.get("member_highs"))},
            "aifs_ens":    {"member_highs": aifs_ens_r.get("member_highs"),
                            "percentiles": _pctls(aifs_ens_r.get("member_highs"))},
        },
        "market": {
            "slug":    poly_r.get("slug"),
            "buckets": [{"label": lbl, "market_prob_pct": prob}
                        for lbl, prob in (poly_r.get("rows") or [])],
        },
        "edges":            edge_data,
        "nws_afd_short_term": afd_r.get("short_term"),
    }


# ─── Main orchestrator ────────────────────────────────────────────────────────

_PRINT_LOCK = __import__("threading").Lock()

def run_location(lat, lon, label, station_id, target_date: str,
                 write_json: bool = False, json_dir: str = ".") -> None:
    tz_name   = TIMEZONES.get(station_id)
    now_utc   = datetime.now(timezone.utc)
    tz        = ZoneInfo(tz_name) if tz_name else timezone.utc
    local_str = now_utc.astimezone(tz).strftime("%H:%M %Z") if tz_name else ""
    utc_str   = now_utc.strftime("%H:%Mz")

    poly_slug = _polymarket_slug(station_id, target_date)

    print(f"\n{label} — {target_date}  local {local_str} {utc_str}")

    # ── Phase 7: concurrent independent fetches ───────────────────────────────
    with ThreadPoolExecutor(max_workers=14) as ex:
        fut_metar    = ex.submit(_fetch_current_metar,  station_id)
        fut_obs      = ex.submit(_fetch_obs_history,    station_id, target_date, tz_name)
        fut_nws      = ex.submit(_fetch_nws,            lat, lon, target_date, now_utc)
        fut_hrrr     = ex.submit(_fetch_hrrr,           lat, lon, target_date, now_utc, tz_name)
        fut_nbm      = ex.submit(_fetch_nbm,            lat, lon, target_date, now_utc, tz_name)
        fut_gfs      = ex.submit(_fetch_gfs,            lat, lon, target_date, now_utc, tz_name)
        fut_ifs_det  = ex.submit(_fetch_ifs_det,        lat, lon, target_date, now_utc, tz_name)
        fut_aifs_det = ex.submit(_fetch_aifs_det,       lat, lon, target_date, now_utc, tz_name)
        fut_ifs_ens  = ex.submit(_fetch_ifs_ens,        lat, lon, target_date, now_utc, tz_name)
        fut_aifs_ens = ex.submit(_fetch_aifs_ens,       lat, lon, target_date, now_utc, tz_name)
        fut_nbm_qmd  = ex.submit(nbm_qmd_high_percentiles,
                                  lat, lon, now_utc, tz_name, target_date)
        fut_poly     = ex.submit(_fetch_polymarket,     poly_slug, label)
        fut_afd      = ex.submit(_fetch_nws_afd,        station_id)

    # Collect results
    metar_r    = fut_metar.result()
    obs_r      = fut_obs.result()
    nws_r      = fut_nws.result()
    hrrr_r     = fut_hrrr.result()
    nbm_r      = fut_nbm.result()
    gfs_r      = fut_gfs.result()
    ifs_det_r  = fut_ifs_det.result()
    aifs_det_r = fut_aifs_det.result()
    ifs_ens_r  = fut_ifs_ens.result()
    aifs_ens_r = fut_aifs_ens.result()
    nbm_qmd_r  = fut_nbm_qmd.result()
    poly_r     = fut_poly.result()
    afd_r      = fut_afd.result()

    # obs_high floor for ensemble (routine METAR only)
    obs_high = obs_r.get("high_routine_f")

    # Print results in defined order
    for line in metar_r["output"]:
        print(line)
    for line in obs_r["output"]:
        print(line)

    print("\n--- Highest temperature predictions for the remainder of the day ---")
    for line in nws_r["output"]:
        print(line)
    for line in hrrr_r["output"]:
        print(line)
    for line in nbm_r["output"]:
        print(line)
    for line in gfs_r["output"]:
        print(line)
    for line in ifs_det_r["output"]:
        print(line)
    for line in aifs_det_r["output"]:
        print(line)

    if nbm_qmd_r:
        for line in nbm_qmd_r["output"]:
            print(line)

    for line in ifs_ens_r["output"]:
        print(line)
    for line in aifs_ens_r["output"]:
        print(line)

    # Post-processed ensemble (sequential; depends on deterministic model vals)
    ens_pp_r = _fetch_ens_pp(
        lat, lon, label, target_date,
        hrrr_r.get("temp_f"), nbm_r.get("temp_f"), gfs_r.get("temp_f"),
        ifs_det_r.get("temp_f"), aifs_det_r.get("temp_f"),
        now_utc, obs_high, tz_name,
    )
    for line in ens_pp_r["output"]:
        print(line)

    # Percentile compare
    nbm_qmd_dm = nbm_qmd_r.get("daily_max") if nbm_qmd_r else None
    _print_pctl_compare(ens_pp_r.get("member_highs"), nbm_qmd_dm,
                        aifs_ens_r.get("member_highs"))

    # Polymarket
    for line in poly_r["output"]:
        print(line)

    # Bucket probabilities
    nbm_bkts  = _nbm_bucket_probs(nbm_qmd_dm, poly_r.get("rows"), obs_high=obs_high)
    aifs_bkts = _ens_bucket_probs(aifs_ens_r.get("member_highs"),
                                   poly_r.get("rows"), obs_high=obs_high)

    edge_lines, edge_data = compare_ensemble_to_market(
        ens_pp_r.get("member_highs"), poly_r.get("rows"),
        nbm_probs=nbm_bkts, aifs_probs=aifs_bkts,
    )
    for line in edge_lines:
        print(line)

    # NWS AFD
    for line in afd_r["output"]:
        print(line)

    # AI summary block
    _print_ai_summary(
        label, station_id, target_date, now_utc, tz_name,
        obs_r, metar_r,
        nws_r, hrrr_r, nbm_r, gfs_r,
        ifs_det_r, aifs_det_r,
        nbm_qmd_r,
        ifs_ens_r, aifs_ens_r, ens_pp_r,
        poly_r, edge_data,
        afd_r,
    )

    # JSON output
    if write_json:
        payload  = _build_json_payload(
            label, station_id, lat, lon, target_date, now_utc, tz_name,
            obs_r, metar_r, nws_r, hrrr_r, nbm_r, gfs_r,
            ifs_det_r, aifs_det_r, nbm_qmd_r,
            ifs_ens_r, aifs_ens_r, ens_pp_r,
            poly_r, edge_data, afd_r,
        )
        fname = (f"{json_dir}/{station_id}_{target_date}_"
                 f"{now_utc.strftime('%H%M')}z.json")
        with open(fname, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        print(f"[JSON]        written → {fname}")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_location(arg: str) -> tuple:
    key = arg.lower()
    if key in PRESETS:
        return PRESETS[key]
    print(f"Unknown location: '{arg}'")
    print(f"Presets: {', '.join(PRESETS.keys())}")
    sys.exit(1)


LOG_PATH = "weathernew.log"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="US weather data collector for Polymarket temperature edge analysis")
    parser.add_argument("locations", nargs="+",
                        help="Station key(s) from presets, or 'all'")
    parser.add_argument("--date", default=None,
                        help="Target date YYYY-MM-DD (default: today in station local time)")
    parser.add_argument("--json", action="store_true",
                        help="Write per-location JSON output files")
    parser.add_argument("--json-dir", default=".", metavar="DIR",
                        help="Directory for JSON files (default: current dir)")
    args = parser.parse_args()

    if len(args.locations) == 1 and args.locations[0].lower() == "all":
        locations = [(*v,) for v in PRESETS.values()]
    else:
        locations = [parse_location(a) for a in args.locations]

    tee = _Tee(LOG_PATH)
    sys.stdout = tee
    try:
        for lat, lon, label, station_id in locations:
            # Determine target date in station's local timezone
            if args.date:
                target_date = args.date
            else:
                tz_name     = TIMEZONES.get(station_id)
                local_tz    = ZoneInfo(tz_name) if tz_name else timezone.utc
                target_date = datetime.now(local_tz).strftime("%Y-%m-%d")

            run_location(lat, lon, label, station_id, target_date,
                         write_json=args.json, json_dir=args.json_dir)
            print()
    finally:
        tee.close()
