import urllib.request
import urllib.parse
import json
import re
import sys
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


class _Tee:
    """Write to both the original stdout and a log file simultaneously."""
    def __init__(self, log_path):
        self._stdout = sys.stdout
        self._log    = open(log_path, "w", encoding="utf-8")

    def write(self, data):
        self._stdout.write(data)
        self._log.write(data)

    def flush(self):
        self._stdout.flush()
        self._log.flush()

    def close(self):
        self._log.close()
        sys.stdout = self._stdout

TARGET_DATE = "2026-04-22"
TEMP_UNIT   = "fahrenheit"
ENS_MODEL   = "ecmwf_ifs025"            # 51-member ECMWF IFS ensemble
AIFS_MODEL  = "ecmwf_aifs025_ensemble"  # ECMWF AIFS ensemble (newer, may be unavailable)

PRESETS = {
    "kord": (41.9742, -87.9073,  "Chicago O'Hare (KORD)",    "KORD", "highest-temperature-in-chicago-on-april-22-2026"),
    "ksea": (47.4499, -122.3118, "Seattle-Tacoma (KSEA)",    "KSEA", "highest-temperature-in-seattle-on-april-22-2026"),
    "kdal": (32.8481, -96.8512,  "Dallas Love Field (KDAL)", "KDAL", "highest-temperature-in-dallas-on-april-22-2026"),
    "klga": (40.7772, -73.8726,  "New York LaGuardia (KLGA)", "KLGA", "highest-temperature-in-nyc-on-april-22-2026"),
    "katl": (33.6407, -84.4277,  "Atlanta (KATL)",            "KATL", "highest-temperature-in-atlanta-on-april-22-2026"),
    "kaus": (30.1975, -97.6664,  "Austin-Bergstrom (KAUS)",   "KAUS", "highest-temperature-in-austin-on-april-22-2026"),
    "khou": (29.6454, -95.2789,  "Houston Hobby (KHOU)",      "KHOU", "highest-temperature-in-houston-on-april-22-2026"),
    "kmia": (25.7959, -80.2870,  "Miami (KMIA)",              "KMIA", "highest-temperature-in-miami-on-april-22-2026"),
}

TIMEZONES = {
    "KORD": "America/Chicago",
    "KSEA": "America/Los_Angeles",
    "KDAL": "America/Chicago",
    "KLGA": "America/New_York",
    "KATL": "America/New_York",
    "KAUS": "America/Chicago",
    "KHOU": "America/Chicago",
    "KMIA": "America/New_York",
}

def _k_to_f(k):
    return (k - 273.15) * 9 / 5 + 32 if k is not None else None

def fetch_json(url, extra_headers=None):
    headers = {"User-Agent": "weathernew.py/1.0"}
    if extra_headers:
        headers.update(extra_headers)
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read())

def fetch_json_params(base, params):
    return fetch_json(f"{base}?{urllib.parse.urlencode(params)}")

# --- NWS ---

def nws_high(lat, lon, target_date, now_utc):
    points_data = fetch_json(f"https://api.weather.gov/points/{lat},{lon}")
    hourly_url = points_data["properties"]["forecastHourly"]
    forecast_data = fetch_json(hourly_url)
    periods = forecast_data["properties"]["periods"]
    raw_upd = forecast_data["properties"].get("updateTime", "")
    upd = raw_upd[:16].replace("T", " ") + "z" if raw_upd else "unknown"

    max_temp, max_unit = None, None
    for p in periods:
        if p["startTime"][:10] == target_date:
            start_utc = datetime.fromisoformat(p["startTime"]).astimezone(timezone.utc)
            if start_utc < now_utc:
                continue  # hour already past
            if max_temp is None or p["temperature"] > max_temp:
                max_temp = p["temperature"]
                max_unit = p["temperatureUnit"]

    if max_temp is None:
        print(f"[NWS]        no data  (updated {upd})")
    else:
        print(f"[NWS]        {max_temp}°{max_unit}  (updated {upd})")

# --- HRRR via Open-Meteo ---

def hrrr_high(lat, lon, target_date, now_utc, obs_high=None, tz_name=None):
    tz = ZoneInfo(tz_name) if tz_name else timezone.utc
    tz_param = tz_name if tz_name else "UTC"
    data = fetch_json_params("https://api.open-meteo.com/v1/forecast", {
        "latitude": lat, "longitude": lon,
        "hourly": "temperature_2m",
        "models": "ncep_hrrr_conus",
        "temperature_unit": TEMP_UNIT,
        "timezone": tz_param,
        "start_date": target_date, "end_date": target_date,
    })
    hourly = data.get("hourly", {})
    cutoff = now_utc.astimezone(tz).strftime("%Y-%m-%dT%H:%M")
    valid = [(t, v) for t, v in zip(hourly.get("time", []), hourly.get("temperature_2m", []))
             if v is not None and t >= cutoff]

    if not valid:
        print("[HRRR]       no data for remainder of day")
        return None
    max_time, max_temp = max(valid, key=lambda x: x[1])
    lo = min(v for _, v in valid)
    peak_label = "local" if tz_name else "z"
    print(f"[HRRR]       {max_temp:.1f}°F  (peak {max_time[-5:]} {peak_label}, range {lo:.1f}–{max_temp:.1f}°F, updated hourly)")
    return max_temp

# --- NBM via Open-Meteo ---

def nbm_high(lat, lon, target_date, now_utc, obs_high=None, tz_name=None):
    tz = ZoneInfo(tz_name) if tz_name else timezone.utc
    tz_param = tz_name if tz_name else "UTC"
    try:
        data = fetch_json_params("https://api.open-meteo.com/v1/forecast", {
            "latitude": lat, "longitude": lon,
            "hourly": "temperature_2m",
            "models": "ncep_nbm_conus",
            "temperature_unit": TEMP_UNIT,
            "timezone": tz_param,
            "start_date": target_date, "end_date": target_date,
        })
    except Exception as e:
        print(f"[NBM]        unavailable ({e})")
        return None
    hourly = data.get("hourly", {})
    cutoff = now_utc.astimezone(tz).strftime("%Y-%m-%dT%H:%M")
    valid = [(t, v) for t, v in zip(hourly.get("time", []), hourly.get("temperature_2m", []))
             if v is not None and t >= cutoff]

    if not valid:
        print("[NBM]        no data for remainder of day")
        return None
    max_time, max_temp = max(valid, key=lambda x: x[1])
    lo = min(v for _, v in valid)
    peak_label = "local" if tz_name else "z"
    print(f"[NBM]        {max_temp:.1f}°F  (peak {max_time[-5:]} {peak_label}, range {lo:.1f}–{max_temp:.1f}°F, updated hourly)")
    return max_temp

# --- NBM QMD percentile high-temperature forecast ---

_QMD_BASE    = "https://nomads.ncep.noaa.gov/pub/data/nccf/com/blend/v5.0"
_QMD_CYCLES  = [0, 6, 12, 18]
_QMD_REGION  = "co"
_QMD_PCTS    = [10, 25, 50, 75, 90]
_QMD_WORKERS = 10
_QMD_TIMEOUT = 60
_qmd_grid_cache: dict[tuple, int] = {}   # {(lat, lon): grid_index}; same CONUS grid for all files


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


def nbm_qmd_high_percentiles(lat: float, lon: float, now_utc: datetime,
                              tz_name: str, target_date: str, obs_high: float | None = None) -> dict[int, float] | None:
    """
    Fetch NBM QMD TMP percentiles for the remaining hours of target_date (local time).
    Returns {10: F, 25: F, 50: F, 75: F, 90: F} where all percentiles are from the hour
    with the highest P50 value (proxy for "hour of peak heating").
    
    LIMITATION: These percentiles reflect intensity-only uncertainty at a single forecast hour,
    not the full range of timing uncertainty. This is a slight underestimate of spread because
    real-world peak heating could occur in adjacent hours with different temperatures.
    """
    try:
        date, cycle = _qmd_latest_cycle()
    except RuntimeError as e:
        print(f"[NBM QMD]    {e}")
        return None
    
    # Warn if cycle is stale (> 8 hours old)
    cycle_age_hours = (now_utc - date.replace(tzinfo=timezone.utc)).total_seconds() / 3600
    if cycle_age_hours > 8:
        print(f"[NBM QMD]    [warn] stale cycle: {cycle_age_hours:.1f} hours old (run {date.strftime('%Y-%m-%d')} {cycle:02d}z)")

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
        print("[NBM QMD]    no remaining forecast hours today")
        return None

    # Step 1: fetch all idx files in parallel
    idx_by_fxx: dict[int, list[str] | None] = {}
    with ThreadPoolExecutor(max_workers=_QMD_WORKERS) as ex:
        futs = {ex.submit(_qmd_fetch_idx, date, cycle, fxx): fxx for fxx in fxx_range}
        for fut in as_completed(futs):
            idx_by_fxx[futs[fut]] = fut.result()

    # Step 2: build per-message download tasks
    tasks: list[tuple[int, int, str, dict]] = []  # (fxx, pct, grib_url, rec)
    for fxx in fxx_range:
        lines = idx_by_fxx.get(fxx)
        if not lines:
            continue
        grib_url, _ = _qmd_urls(date, cycle, fxx)
        for pct, rec in _qmd_parse_tmp_recs(lines).items():
            tasks.append((fxx, pct, grib_url, rec))

    # Step 3: fetch all GRIB bytes in parallel
    raw_by_key: dict[tuple[int, int], bytes | None] = {}
    with ThreadPoolExecutor(max_workers=_QMD_WORKERS) as ex:
        futs = {ex.submit(_qmd_fetch_bytes, grib_url, rec): (fxx, pct)
                for fxx, pct, grib_url, rec in tasks}
        for fut in as_completed(futs):
            raw_by_key[futs[fut]] = fut.result()

    # Step 4: decode sequentially (eccodes not thread-safe); collect per-pct raw Kelvin values
    # and track per-fxx values to find peak P50 hour
    vals_by_pct: dict[int, list[float]] = {p: [] for p in _QMD_PCTS}
    vals_by_fxx_pct: dict[tuple[int, int], float] = {}  # (fxx, pct) -> Kelvin value
    for (fxx, pct), raw in raw_by_key.items():
        if raw is None:
            continue
        v = _qmd_decode(raw, lat, lon)
        if v is not None:
            vals_by_pct[pct].append(v)
            vals_by_fxx_pct[(fxx, pct)] = v

    # Step 5: find the hour (fxx) with highest P50, then return all percentiles from that hour
    # This is "hour of peak heating" percentiles, not max-per-percentile
    peak_p50_fxx = None
    peak_p50_val = -273.15  # absolute zero in Kelvin
    for fxx in fxx_range:
        if (fxx, 50) in vals_by_fxx_pct:
            val = vals_by_fxx_pct[(fxx, 50)]
            if val > peak_p50_val:
                peak_p50_val = val
                peak_p50_fxx = fxx

    result: dict[int, float] = {}
    if peak_p50_fxx is not None:
        for p in _QMD_PCTS:
            if (peak_p50_fxx, p) in vals_by_fxx_pct:
                result[p] = _k_to_f(vals_by_fxx_pct[(peak_p50_fxx, p)])

    if len(result) < 3:
        print("[NBM QMD]    insufficient data")
        return None

    run_str    = f"{date.strftime('%Y-%m-%d')} {cycle:02d}z"
    pct_str    = "  ".join(f"{p}th:{result[p]:.1f}" for p in _QMD_PCTS if p in result)
    first_valid = (init_utc + timedelta(hours=fxx_range[0])).astimezone(ZoneInfo(tz_name))
    last_valid  = (init_utc + timedelta(hours=fxx_range[-1])).astimezone(ZoneInfo(tz_name))
    window_str  = (f"{first_valid.strftime('%I%p').lstrip('0').lower()}–"
                   f"{last_valid.strftime('%I%p').lstrip('0').lower()}")
    
    # Note: these are "hour of peak heating" percentiles, not "daily max" percentiles
    peak_p50_hour_utc = (init_utc + timedelta(hours=peak_p50_fxx)).astimezone(ZoneInfo(tz_name))
    peak_time_str = peak_p50_hour_utc.strftime("%I%p").lstrip("0").lower()
    print(f"[NBM QMD]    {pct_str}°F  (hour of peak heating {peak_time_str} local, run {run_str}, window {window_str})")
    print(f"             [intensity-only uncertainty; timing uncertainty not included]")
    
    # Diagnostic: if obs_high floors the distribution meaningfully, note it
    if obs_high is not None:
        # Count how many percentiles are floored
        floored = [p for p in _QMD_PCTS if result[p] < obs_high]
        if len(floored) >= 3:  # half or more — material shift
            highest_floored = max(floored)
            print(f"  [note] obs_high {obs_high:.1f}°F floors QMD P{highest_floored} and below; "
                  f"effective distribution collapses toward upper tail")
        elif obs_high > result[50]:
            print(f"  [note] obs_high {obs_high:.1f}°F above QMD P50 ({result[50]:.1f}°F); "
                  f"distribution moderately shifted by floor")
    
    return result


# --- ECMWF IFS deterministic ---

def ecmwf_deterministic_high(lat, lon, target_date, now_utc, obs_high=None, tz_name=None):
    tz = ZoneInfo(tz_name) if tz_name else timezone.utc
    tz_param = tz_name if tz_name else "UTC"
    data = fetch_json_params("https://api.open-meteo.com/v1/ecmwf", {
        "latitude": lat, "longitude": lon,
        "hourly": "temperature_2m",
        "temperature_unit": TEMP_UNIT,
        "timezone": tz_param,
        "start_date": target_date, "end_date": target_date,
    })
    hourly = data["hourly"]
    times = hourly.get("time", [])
    model_run = times[0].replace("T", " ") if times else "unknown"
    cutoff = now_utc.astimezone(tz).strftime("%Y-%m-%dT%H:%M")
    vals = [v for t, v in zip(times, hourly.get("temperature_2m", []))
            if v is not None and t >= cutoff]

    if not vals:
        print("[ECMWF IFS]  no data for remainder of day")
        return None
    max_val = max(vals)
    print(f"[ECMWF IFS]  {max_val:.1f}°F  (model run {model_run})")
    return max_val

# --- Shared ensemble stats printer ---

def _weighted_percentile(values, weights, percentiles):
    """Weighted quantiles via linear interpolation on the empirical CDF."""
    idx         = np.argsort(values)
    sorted_vals = values[idx]
    cumw        = np.cumsum(weights[idx])
    cumw       /= cumw[-1]
    return np.interp([p / 100 for p in percentiles], cumw, sorted_vals)


def _print_ens_stats(member_highs, weights=None):
    maxes = np.array(member_highs)

    if weights is not None:
        w        = np.array(weights)
        w        = w / w.sum()
        mean_val = float(np.average(maxes, weights=w))
        std_val  = float(np.sqrt(np.average((maxes - mean_val) ** 2, weights=w)))
        p        = _weighted_percentile(maxes, w, [5, 10, 25, 50, 75, 90, 95])
    else:
        w        = None
        mean_val = float(np.mean(maxes))
        std_val  = float(np.std(maxes))
        p        = np.percentile(maxes, [5, 10, 25, 50, 75, 90, 95])

    pct_labels = ["5th", "10th", "25th", "50th", "75th", "90th", "95th"]
    pct_str = "  ".join(f"{l}:{v:.1f}" for l, v in zip(pct_labels, p))
    print(f"  Mean {mean_val:.1f}  Median {p[3]:.1f}  Std {std_val:.1f}  Range {np.min(maxes):.1f}–{np.max(maxes):.1f}°F")
    print(f"  {pct_str}°F")

    bar_max = 24
    if w is not None:
        wcounts, edges = np.histogram(maxes, bins=10, weights=w)
        print(f"\n  Distribution ({len(member_highs)} members, weighted):")
        peak = wcounts.max()
        for i in range(len(wcounts)):
            lo, hi = edges[i], edges[i + 1]
            bar = "█" * int(wcounts[i] / peak * bar_max) if peak > 0 else ""
            print(f"  {lo:>5.1f}–{hi:<5.1f}°F  {wcounts[i]*100:>4.1f}%  {bar}")
    else:
        counts, edges = np.histogram(maxes, bins=10)
        print(f"\n  Distribution ({len(member_highs)} members):")
        for i in range(len(counts)):
            lo, hi = edges[i], edges[i + 1]
            bar = "█" * int(counts[i] / counts.max() * bar_max)
            print(f"  {lo:>5.1f}–{hi:<5.1f}°F  {counts[i]:>3}  {bar}")

    p5_t  = round(float(p[0]))
    p95_t = round(float(p[6]))
    thresholds = list(range(p5_t, p95_t + 1, 5))
    probs = []
    for t in thresholds:
        if w is not None:
            prob = float(np.sum(w[maxes >= t])) * 100
        else:
            prob = float(np.mean(maxes >= t)) * 100
        probs.append(f"P(>={t}) {prob:.0f}%")
    if probs:
        print(f"\n  " + "  ".join(probs))


# --- ECMWF IFS Ensemble (ecmwf_ifs025, 51 members) ---

def ecmwf_ensemble_high(lat, lon, label, target_date, now_utc, tz_name=None):
    tz = ZoneInfo(tz_name) if tz_name else timezone.utc
    tz_param = tz_name if tz_name else "UTC"
    data = fetch_json_params("https://ensemble-api.open-meteo.com/v1/ensemble", {
        "latitude": lat, "longitude": lon,
        "hourly": "temperature_2m",
        "models": ENS_MODEL,
        "start_date": target_date, "end_date": target_date,
        "temperature_unit": TEMP_UNIT,
        "timezone": tz_param,
    })
    hourly = data.get("hourly", {})

    # First timestamp in the UTC hourly data is the model run init time
    times = hourly.get("time", [])
    model_run = times[0].replace("T", " ") if times else "unknown"
    cutoff = now_utc.astimezone(tz).strftime("%Y-%m-%dT%H:%M")

    member_keys = sorted(k for k in hourly if k.startswith("temperature_2m_member"))
    if "temperature_2m" in hourly and "temperature_2m" not in member_keys:
        member_keys = ["temperature_2m"] + member_keys

    member_highs = []
    for key in member_keys:
        vals = [v for t, v in zip(times, hourly.get(key, [])) if v is not None and t >= cutoff]
        if vals:
            member_highs.append(max(vals))

    print(f"\n[ECMWF ENS]  {label} — {target_date}  ({len(member_highs)} members, model run {model_run})")
    if not member_highs:
        print("  No data returned")
        return
    _print_ens_stats(member_highs)


# --- ECMWF AIFS (commented out) ---

# def aifs_deterministic_high(lat, lon, label, target_date): ...
# def aifs_ensemble_high(lat, lon, label, target_date): ...

# --- ECMWF AIFS (commented out) ---

# def aifs_deterministic_high(lat, lon, label, target_date): ...
# def aifs_ensemble_high(lat, lon, label, target_date): ...

# --- Post-processed ensemble (bias-corrected + spread-inflated) ---

def ecmwf_ensemble_postprocessed(lat, lon, label, target_date, hrrr_val, nbm_val, ifs_val=None, now_utc=None, obs_high=None, tz_name=None):
    """
    Shift the raw ENS distribution to match a freshness-weighted anchor, then scale
    spread using a two-component variance model: ensemble under-dispersion correction
    plus anchor uncertainty (stdev across fresh point forecasts).
    """
    if all(v is None for v in [hrrr_val, nbm_val, ifs_val]):
        print(f"\n[ECMWF ENS PP]  {label} — skipped (no fresh anchor available)")
        return None

    tz = ZoneInfo(tz_name) if tz_name else timezone.utc
    tz_param = tz_name if tz_name else "UTC"

    try:
        data = fetch_json_params("https://ensemble-api.open-meteo.com/v1/ensemble", {
            "latitude": lat, "longitude": lon,
            "hourly": "temperature_2m",
            "models": ENS_MODEL,
            "start_date": target_date, "end_date": target_date,
            "temperature_unit": TEMP_UNIT,
            "timezone": tz_param,
        })
    except Exception as e:
        print(f"\n[ECMWF ENS PP]  {label} — unavailable ({e})")
        return None

    hourly = data.get("hourly", {})
    times = hourly.get("time", [])
    cutoff = now_utc.astimezone(tz).strftime("%Y-%m-%dT%H:%M") if now_utc is not None else ""

    # Step 5: Include the IFS control run (temperature_2m) alongside perturbed members
    member_keys = sorted(k for k in hourly if k.startswith("temperature_2m_member"))
    if "temperature_2m" in hourly and "temperature_2m" not in member_keys:
        member_keys = ["temperature_2m"] + member_keys

    raw_highs = []
    for key in member_keys:
        vals = [v for t, v in zip(times, hourly.get(key, [])) if v is not None and t >= cutoff]
        if vals:
            raw_highs.append(max(vals))

    if not raw_highs:
        print(f"\n[ECMWF ENS PP]  {label} — no data returned")
        return None
    if len(raw_highs) < 50:
        print(f"  [warn] only {len(raw_highs)} members found (expected ≥50)")

    # Step 4: Freshness-adjusted weighted anchor
    # IFS age estimated from ensemble model run time (inferred from times[0], which is an approximation)
    # Ideally we'd get the actual model run timestamp from metadata, but Open-Meteo doesn't expose it directly
    ifs_age_hours_approx = 9.0  # conservative fallback
    if times and now_utc is not None:
        try:
            run_dt_approx = datetime.fromisoformat(times[0]).astimezone(timezone.utc)
            ifs_age_hours_approx = max(0.0, (now_utc - run_dt_approx).total_seconds() / 3600)
        except Exception:
            print(f"  [warn] could not parse IFS model run time approximation from '{times[0] if times else ''}'; using fallback age {ifs_age_hours_approx:.0f}h")

    _anchor_candidates = [
        ("HRRR", hrrr_val, 0.7, 0.0),
        ("NBM",  nbm_val,  1.0, 0.0),
        ("IFS",  ifs_val,  0.4, ifs_age_hours_approx),
    ]
    anchor_weights = {
        name: w_base * float(np.exp(-age / 18))
        for name, val, w_base, age in _anchor_candidates if val is not None
    }
    anchor_vals = {
        name: val
        for name, val, w_base, age in _anchor_candidates if val is not None
    }
    total_w = sum(anchor_weights.values())
    anchor = sum(anchor_weights[k] * anchor_vals[k] for k in anchor_weights) / total_w

    raw = np.array(raw_highs)
    ens_mean = float(np.mean(raw))
    delta    = anchor - ens_mean

    # Two-component variance model
    sigma_ens = float(np.std(raw))
    anchor_pts = list(anchor_vals.values())
    sigma_anchor = max(float(np.std(anchor_pts)) if len(anchor_pts) >= 2 else 0.0, 0.5)
    sigma_ens_corrected = sigma_ens * 1.15
    sigma_target = float(np.sqrt(sigma_ens_corrected ** 2 + sigma_anchor ** 2))
    scale = sigma_target / sigma_ens if sigma_ens > 0 else 1.0

    SCALE_HYBRID_THRESHOLD = 1.5
    # Shift entire distribution to anchor, then scale spread around new mean
    shifted = raw + delta
    if scale <= SCALE_HYBRID_THRESHOLD:
        inflated = anchor + (shifted - anchor) * scale
    else:
        # Hybrid: cap multiplicative scale, then add independent Gaussian noise
        # to cover the remaining variance gap without over-stretching members.
        capped_scale = SCALE_HYBRID_THRESHOLD
        inflated = anchor + (shifted - anchor) * capped_scale
        sigma_residual = float(np.sqrt(max(sigma_target ** 2 - (sigma_ens * capped_scale) ** 2, 0.0)))
        seed = hash(f"{label}_{target_date}") & 0xFFFFFFFF
        rng = np.random.default_rng(seed)
        inflated = inflated + rng.normal(0.0, sigma_residual, size=inflated.shape)
        print(f"  [note] scale={scale:.2f}x > {SCALE_HYBRID_THRESHOLD} — hybrid correction applied: "
              f"multiplicative cap={capped_scale}x + additive σ_extra={sigma_residual:.2f}°F")

    # Warn if all raw point forecasts are below obs_high (daily high locked at obs)
    if obs_high is not None:
        raw_pts = [v for v in [hrrr_val, nbm_val, ifs_val] if v is not None]
        if raw_pts and all(v < obs_high for v in raw_pts):
            print(f"  [note] all model forecasts ({', '.join(f'{v:.1f}' for v in raw_pts)}°F) below "
                  f"obs_high ({obs_high:.1f}°F); daily high effectively locked at obs_high")
        inflated = np.maximum(inflated, obs_high)

    weight_str = "  ".join(f"{k}:{anchor_weights[k]:.2f}" for k in anchor_weights)
    print(f"\n[ECMWF ENS PP]  {label} — {target_date}  "
          f"({len(raw_highs)} members, anchor={anchor:.1f}°F [weights: {weight_str}], "
          f"delta={delta:+.1f}°F, σ_ens={sigma_ens:.2f}°F, σ_anchor={sigma_anchor:.2f}°F, "
          f"scale={scale:.2f}x)")
    _print_ens_stats(list(inflated))
    return inflated


# --- Current temperature from latest METAR ---

def _metar_obs_utc(ob):
    """Return the true observation time as a UTC-aware datetime.
    Prefers obsTime (Unix epoch, unrounded) over reportTime (rounded to hour)."""
    epoch = ob.get("obsTime")
    if epoch is not None:
        return datetime.fromtimestamp(int(epoch), tz=timezone.utc)
    ts = ob.get("reportTime", "")
    if ts:
        return datetime.fromisoformat(ts).astimezone(timezone.utc)
    return None


def fetch_current_metar(station_id):
    """Returns (temp_F, time_str) from the latest ASOS/METAR observation, or (None, None) on failure.
    Priority: NWS observations (5-minute ASOS, no token) → aviationweather.gov."""

    # NWS observations — includes 5-minute ASOS readings between METARs, no token required
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
            return temp_f, obs_utc.strftime("%H:%Mz")
    except Exception as e:
        print(f"  [NWS obs fetch failed: {e}]")

    # Last resort: aviationweather.gov (METAR only, ~hourly)
    try:
        data = fetch_json(
            f"https://aviationweather.gov/api/data/metar?ids={station_id}&format=json"
        )
        if not data:
            return None, None
        ob      = data[0]
        temp_c  = ob.get("temp")
        obs_utc = _metar_obs_utc(ob)
        if temp_c is None or obs_utc is None:
            return None, None
        temp_f = temp_c * 9 / 5 + 32
        return temp_f, obs_utc.strftime("%H:%Mz")
    except Exception as e:
        print(f"  [METAR fetch failed: {e}]")
    return None, None


def fetch_observed_high_today(station_id, target_date, tz_name, routine_only=False):
    """Returns (max_temp_F, local_time_str, utc_time_str) for the observed high today, or (None, None, None).
    If routine_only=True, SPECIs are excluded.
    Uses aviationweather.gov."""

    # aviationweather.gov
    try:
        data = fetch_json(
            f"https://aviationweather.gov/api/data/metar?ids={station_id}&format=json&hours=24"
        )
        if not data:
            return None, None, None
        tz = ZoneInfo(tz_name) if tz_name else timezone.utc
        best_temp  = None
        best_local = None
        best_utc   = None
        for ob in data:
            if routine_only and ob.get("metarType", "METAR") == "SPECI":
                continue
            temp_c  = ob.get("temp")
            obs_utc = _metar_obs_utc(ob)
            if temp_c is None or obs_utc is None:
                continue
            obs_local = obs_utc.astimezone(tz)
            if obs_local.strftime("%Y-%m-%d") != target_date:
                continue
            temp_f = temp_c * 9 / 5 + 32
            if best_temp is None or temp_f > best_temp:
                best_temp  = temp_f
                best_local = obs_local.strftime("%H:%M %Z")
                best_utc   = obs_utc.strftime("%H:%Mz")
        return best_temp, best_local, best_utc
    except Exception:
        return None, None, None


# --- Polymarket prediction market odds ---

def polymarket_odds(slug, label):
    """Fetch and display Polymarket odds for the given event slug."""
    try:
        data = fetch_json(f"https://gamma-api.polymarket.com/events?slug={slug}")
    except Exception as e:
        print(f"\n[POLYMARKET]  unavailable ({e})")
        return None

    if not data:
        print(f"\n[POLYMARKET]  no event found for slug: {slug}")
        return None

    event = data[0] if isinstance(data, list) else data
    markets = event.get("markets", [])
    if not markets:
        print(f"\n[POLYMARKET]  no markets found")
        return None

    # Extract range label and "Yes" probability from each market
    rows = []
    for m in markets:
        question = m.get("question", "")
        prices = m.get("outcomePrices", "[]")
        if isinstance(prices, str):
            try:
                prices = json.loads(prices)
            except json.JSONDecodeError:
                continue
        try:
            yes_prob = float(prices[0]) * 100
        except (IndexError, ValueError, TypeError):
            continue
        # Pull the temperature range out of the question text
        match = re.search(r"be (.+?)\s+on\s+April", question, re.IGNORECASE)
        range_label = re.sub(r"^between\s+", "", match.group(1) if match else question)
        rows.append((range_label, yes_prob))

    fetched_at = datetime.now(timezone.utc).strftime("%H:%Mz")
    print(f"\n[POLYMARKET]  {label}  (fetched {fetched_at})")

    bar_max = 20
    peak = max((p for _, p in rows), default=1)
    for range_label, prob in rows:
        bar = "█" * int(prob / peak * bar_max) if peak > 0 else ""
        print(f"  {range_label:<22}  {prob:>5.1f}%  {bar}")
    return rows


# --- NBM distribution fitting + bucket probabilities ---

def _fit_nbm_distribution(nbm_pctls: dict[int, float]):
    """
    Fit skew-normal to the five NBM percentile values.
    Returns a frozen scipy distribution, or None on failure.
    Falls back to Gaussian if skew-normal is poorly constrained.
    """
    probs  = [p / 100 for p in _QMD_PCTS]
    target = np.array([nbm_pctls[p] for p in _QMD_PCTS])

    med    = nbm_pctls[50]
    iqr    = nbm_pctls[75] - nbm_pctls[25]
    scale0 = max(iqr / 1.349, 0.5)   # IQR of normal ≈ 1.349σ

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

    # Gaussian fallback
    sigma = (nbm_pctls[90] - nbm_pctls[10]) / 2.564   # P10–P90 span ≈ 2.564σ
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

    floor       = obs_high if obs_high is not None else -np.inf
    p_above     = 1.0 - dist.cdf(floor)
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


# --- Phase 2: ENS PP vs NBM percentile diagnostic ---

def _print_pctl_compare(ens_members, nbm_pctls: dict[int, float] | None) -> None:
    if ens_members is None or nbm_pctls is None:
        return

    pcts     = [10, 25, 50, 75, 90]
    ens_vals = np.percentile(np.array(ens_members), pcts)
    nbm_vals = [nbm_pctls.get(p) for p in pcts]

    if any(v is None for v in nbm_vals):
        return

    col = 7
    hdr = "             " + "".join(f"  {str(p)+'th':>{col}}" for p in pcts)
    print(f"\n[PCTL COMPARE]")
    print(hdr)
    print("  ENS PP    " + "".join(f"  {v:>{col}.1f}" for v in ens_vals))
    print("  NBM       " + "".join(f"  {v:>{col}.1f}" for v in nbm_vals))
    deltas = [float(e) - float(n) for e, n in zip(ens_vals, nbm_vals)]
    print("  Δ         " + "".join(f"  {d:>+{col}.1f}" for d in deltas))

    tail_warn = abs(deltas[0]) > 1.5 or abs(deltas[-1]) > 1.5
    if tail_warn:
        print("  [WARN] ENS-NBM percentile disagreement > 1.5°F at tails"
              " — ENS spread may be miscalibrated")


# --- Ensemble vs market comparison ---

def compare_ensemble_to_market(inflated_members, market_rows,
                                nbm_probs: list[float] | None = None):
    """
    Side-by-side bucket probability table: ENS PP vs NBM (if available) vs Market.
    Phase 4: flags buckets where both sources agree in direction and |edge| >= 5pp.
    """
    if market_rows is None or inflated_members is None or len(inflated_members) == 0:
        return
    members  = np.array(inflated_members)
    has_nbm  = nbm_probs is not None and len(nbm_probs) == len(market_rows)

    if has_nbm:
        print(f"\n{'Bucket':<24}  {'ENS':>8}  {'NBM':>8}  {'Market':>8}  {'Edge(ENS)':>10}  {'Edge(NBM)':>10}")
        print(f"  {'─'*22}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*10}  {'─'*10}")
    else:
        print(f"\n{'Bucket':<24}  {'Ensemble':>8}  {'Market':>8}  {'Edge':>8}")
        print(f"  {'─'*22}  {'─'*8}  {'─'*8}  {'─'*8}")

    agreed: list[tuple[str, float, float]] = []

    for i, (label, mkt_prob) in enumerate(market_rows):
        m_below = re.match(r"(\d+\.?\d*)°?F?\s+or\s+below",  label, re.IGNORECASE)
        m_range = re.match(r"(\d+\.?\d*)\s*[-–]\s*(\d+\.?\d*)°?F?", label, re.IGNORECASE)
        m_above = re.match(r"(\d+\.?\d*)°?F?\s+or\s+higher", label, re.IGNORECASE)

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

        nbm_p = nbm_probs[i] if has_nbm else float("nan")

        if np.isnan(ens_prob):
            if has_nbm:
                print(f"  {label:<22}  {'n/a':>8}  {'n/a':>8}  {mkt_prob:>7.1f}%  {'n/a':>10}  {'n/a':>10}")
            else:
                print(f"  {label:<22}  {'n/a':>8}  {mkt_prob:>7.1f}%  {'n/a':>8}")
            continue

        ens_edge = ens_prob - mkt_prob

        if has_nbm and not np.isnan(nbm_p):
            nbm_edge = nbm_p - mkt_prob
            flag = ""
            if (np.sign(ens_edge) == np.sign(nbm_edge)
                    and abs(ens_edge) >= 5 and abs(nbm_edge) >= 5):
                flag = "  ★"
                agreed.append((label, ens_edge, nbm_edge))
            print(f"  {label:<22}  {ens_prob:>7.1f}%  {nbm_p:>7.1f}%  {mkt_prob:>7.1f}%"
                  f"  {ens_edge:>+9.1f}pp  {nbm_edge:>+9.1f}pp{flag}")
        else:
            if has_nbm:
                print(f"  {label:<22}  {ens_prob:>7.1f}%  {'n/a':>8}  {mkt_prob:>7.1f}%"
                      f"  {ens_edge:>+9.1f}pp  {'n/a':>10}")
            else:
                print(f"  {label:<22}  {ens_prob:>7.1f}%  {mkt_prob:>7.1f}%  {ens_edge:>+7.1f}pp")

    # Phase 4: agreement summary
    if agreed:
        print(f"\n  ★ Both sources agree (|edge| ≥ 5pp, same direction):")
        for lbl, ens_e, nbm_e in agreed:
            direction = "OVER" if ens_e > 0 else "UNDER"
            print(f"    {lbl}: {direction}  ENS {ens_e:+.1f}pp  NBM {nbm_e:+.1f}pp")


# --- CLI ---

def parse_location(arg):
    key = arg.lower()
    if key in PRESETS:
        return PRESETS[key]
    print(f"Unknown location: '{arg}'")
    print(f"Presets: {', '.join(PRESETS.keys())}")
    sys.exit(1)

def run_location(lat, lon, label, station_id, poly_slug=None):
    tz_name   = TIMEZONES.get(station_id)
    now_utc   = datetime.now(timezone.utc)
    local_str = now_utc.astimezone(ZoneInfo(tz_name)).strftime("%H:%M %Z") if tz_name else ""
    utc_str   = now_utc.strftime("%H:%Mz")
    print(f"\n{label} — {TARGET_DATE}  local {local_str} {utc_str}")
    temp_f, ts = fetch_current_metar(station_id)
    if temp_f is not None:
        print(f"[OBS CURRENT]{temp_f:.1f}°F  current temperature  (updated {ts})")
    else:
        print(f"[OBS CURRENT]no data")
    obs_high_true, obs_local_true, obs_utc_true = fetch_observed_high_today(station_id, TARGET_DATE, tz_name, routine_only=False)
    obs_high,      obs_local_time, obs_utc_time = fetch_observed_high_today(station_id, TARGET_DATE, tz_name, routine_only=True)
    if obs_high_true is not None:
        print(f"[OBS HIGH]   {obs_high_true:.1f}°F  observed high today (all obs)  ({obs_local_true} / {obs_utc_true})")
    else:
        print(f"[OBS HIGH]   no data")
    if obs_high is not None:
        suffix = f"  ← ENS floor" if obs_high != obs_high_true else ""
        print(f"[OBS ROUTINE]{obs_high:.1f}°F  routine METARs only  ({obs_local_time} / {obs_utc_time}){suffix}")
    else:
        print(f"[OBS ROUTINE] no data")
    print()
    print("--- Highest temperature predictions for the remainder of the day ---")
    nws_high(lat, lon, TARGET_DATE, now_utc)
    hrrr_val    = hrrr_high(lat, lon, TARGET_DATE, now_utc, obs_high=obs_high, tz_name=tz_name)
    nbm_val     = nbm_high(lat, lon, TARGET_DATE, now_utc, obs_high=obs_high, tz_name=tz_name)
    nbm_pctls   = nbm_qmd_high_percentiles(lat, lon, now_utc, tz_name, TARGET_DATE, obs_high=obs_high)
    ifs_val     = ecmwf_deterministic_high(lat, lon, TARGET_DATE, now_utc, obs_high=obs_high, tz_name=tz_name)
    inflated_members = ecmwf_ensemble_postprocessed(
        lat, lon, label, TARGET_DATE, hrrr_val, nbm_val, ifs_val, now_utc, obs_high=obs_high, tz_name=tz_name)
    _print_pctl_compare(inflated_members, nbm_pctls)
    market_rows = None
    if poly_slug:
        market_rows = polymarket_odds(poly_slug, label)
    nbm_bkts = _nbm_bucket_probs(nbm_pctls, market_rows, obs_high=obs_high)
    compare_ensemble_to_market(inflated_members, market_rows, nbm_probs=nbm_bkts)


LOG_PATH = "weathernew.log"

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python weathernew.py <location> [location ...]")
        print(f"Presets: {', '.join(PRESETS.keys())}, all")
        sys.exit(1)

    args = sys.argv[1:]
    if len(args) == 1 and args[0].lower() == "all":
        locations = [(*v,) for v in PRESETS.values()]
    else:
        locations = [parse_location(a) for a in args]

    tee = _Tee(LOG_PATH)
    sys.stdout = tee
    try:
        for lat, lon, label, station_id, poly_slug in locations:
            run_location(lat, lon, label, station_id, poly_slug)
            print()
    finally:
        tee.close()
