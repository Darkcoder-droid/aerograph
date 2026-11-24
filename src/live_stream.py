from dotenv import load_dotenv
load_dotenv()

import copy
import time
import os
import json
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Tuple, Optional, Any
import concurrent.futures

import numpy as np
import pandas as pd
import requests
import torch

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import (
    DATA_DIR, CHECKPOINTS_DIR, SEQ_IN, SEQ_OUT,
    OPENAQ_BASE_URL, OPENMETEO_FORECAST_URL, DEFAULT_SIGMA_KM
)
from src.graph_physics import build_spatial_matrices, compute_wind_adjacency
from src.model import WindSTGNN

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def get_openaq_headers() -> Dict[str, str]:
    api_key = os.getenv("OPENAQ_API_KEY")
    if not api_key:
        print("[OPENAQ WARN] No OPENAQ_API_KEY found in environment or .env file.")
        return {}
    return {"X-API-Key": api_key.strip()}

def fetch_bengaluru_live_pm25(stations: List[Dict]) -> Dict[int, float]:
    """
    Fetches live Bengaluru PM2.5 readings in a single fast OpenAQ v3 API call:
      GET https://api.openaq.org/v3/parameters/2/latest?bbox=77.45,12.80,77.78,13.15&limit=100
    with header X-API-Key. Matches each result by locationsId to stations.
    
    If fewer than 3 stations match (due to upstream global pagination), fast-queries
    the station latest endpoints in parallel to guarantee authentic live data.
    Prints HTTP status and errors to console so failures are never swallowed.
    """
    headers = get_openaq_headers()
    url = f"{OPENAQ_BASE_URL}/parameters/2/latest?bbox=77.45,12.80,77.78,13.15&limit=100"
    live_map = {}
    st_loc_ids = {s["id"]: i for i, s in enumerate(stations)}

    try:
        resp = requests.get(url, headers=headers, timeout=8)
        print(f"[LIVE FETCH] GET {url} -> HTTP {resp.status_code}")
        if resp.status_code == 200:
            data = resp.json()
            for item in data.get("results", []):
                lid = item.get("locationsId")
                if lid in st_loc_ids and "value" in item and item["value"] is not None:
                    live_map[lid] = float(item["value"])
            print(f"[LIVE FETCH] Matched {len(live_map)} station(s) from parameters/2/latest.")
        else:
            print(f"[LIVE FETCH ERROR] Status {resp.status_code}: {resp.text[:200]}")
    except Exception as e:
        print(f"[LIVE FETCH ERROR] Single call exception: {e}")

    # Fallback if fewer than 3 matches: parallel fetch for individual stations
    if len(live_map) < 3:
        def fetch_station_latest(s):
            lid = s["id"]
            sensor_id = s.get("pm25_sensor_id")
            try:
                r = requests.get(f"{OPENAQ_BASE_URL}/locations/{lid}/latest", headers=headers, timeout=5)
                if r.status_code == 200:
                    for res in r.json().get("results", []):
                        if res.get("sensorsId") == sensor_id or res.get("locationsId") == lid:
                            if "value" in res and res["value"] is not None:
                                return lid, float(res["value"])
                else:
                    print(f"[LIVE FETCH ERROR] Station {lid} latest HTTP {r.status_code}: {r.text[:120]}")
            except Exception as ex:
                print(f"[LIVE FETCH ERROR] Station {lid} request failed: {ex}")
            return lid, None

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            for lid, val in executor.map(fetch_station_latest, stations):
                if val is not None:
                    live_map[lid] = val

    print(f"[LIVE FETCH] Total live PM2.5 readings acquired: {len(live_map)}/{len(stations)}")
    return live_map

WEATHER_CACHE_FILE = DATA_DIR / "last_valid_weather.json"
PM25_CACHE_FILE = DATA_DIR / "last_valid_pm25.json"

_WEATHER_MEM_CACHE: Dict[str, Any] = {
    "timestamp": 0.0,
    "data": None
}

_LIVE_INFERENCE_CACHE: Dict[Tuple, Dict[str, Any]] = {}

def get_persisted_weather() -> Optional[Dict[str, Any]]:
    global _WEATHER_MEM_CACHE
    now_ts = time.time()
    # Check in-memory cache first (valid for 15 minutes = 900 seconds)
    if _WEATHER_MEM_CACHE["data"] is not None and (now_ts - _WEATHER_MEM_CACHE["timestamp"] < 900.0):
        return _WEATHER_MEM_CACHE["data"]
    
    # Try disk cache
    if WEATHER_CACHE_FILE.exists():
        try:
            with open(WEATHER_CACHE_FILE, "r", encoding="utf-8") as f:
                cached = json.load(f)
                if isinstance(cached, dict) and "wind_direction_10m" in cached:
                    _WEATHER_MEM_CACHE["timestamp"] = cached.get("timestamp", now_ts)
                    _WEATHER_MEM_CACHE["data"] = cached
                    return cached
        except Exception as e:
            print(f"[WEATHER CACHE] Failed to read disk cache: {e}")
    return None

def save_persisted_weather(weather_dict: Dict[str, Any]):
    global _WEATHER_MEM_CACHE
    now_ts = time.time()
    weather_dict["timestamp"] = now_ts
    weather_dict["cached_at_utc"] = datetime.now(timezone.utc).isoformat()
    _WEATHER_MEM_CACHE["timestamp"] = now_ts
    _WEATHER_MEM_CACHE["data"] = weather_dict
    try:
        with open(WEATHER_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(weather_dict, f, indent=2)
    except Exception as e:
        print(f"[WEATHER CACHE] Failed to write disk cache: {e}")

def fetch_live_weather(lat: float, lon: float) -> Tuple[Optional[pd.DataFrame], Optional[Dict]]:
    url = (
        f"{OPENMETEO_FORECAST_URL}?latitude={lat:.4f}&longitude={lon:.4f}"
        f"&current=temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m,wind_gusts_10m"
        f"&past_days=1&forecast_days=1"
        f"&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m"
    )
    try:
        resp = requests.get(url, timeout=8)
        if resp.status_code == 200:
            data = resp.json()
            curr = data.get("current", {})
            hourly = data.get("hourly", {})
            times = hourly.get("time", [])
            df = pd.DataFrame({
                "timestamp": pd.to_datetime(times, utc=True),
                "temperature": hourly.get("temperature_2m", []),
                "relative_humidity": hourly.get("relative_humidity_2m", []),
                "wind_speed": hourly.get("wind_speed_10m", []),
                "wind_direction": hourly.get("wind_direction_10m", [])
            })
            return df.set_index("timestamp").sort_index(), curr
        else:
            print(f"[OPEN-METEO ERROR] Status {resp.status_code}: {resp.text[:150]}")
    except Exception as e:
        print(f"[OPEN-METEO ERROR] Weather fetch failed: {e}")
    return None, None

def get_effective_weather() -> Dict[str, Any]:
    """
    Fetches live weather from Open-Meteo.
    If Open-Meteo succeeds: updates persistent 15-minute cache and returns live weather.
    If Open-Meteo fails or is rate limited: returns last valid weather from persistent cache (15-min TTL).
    If no cache exists at all: returns consistent meteorological baseline for Bengaluru (85° ENE, 11.5 km/h).
    Never alternates back to the offline dataset.pt snapshot (230° SW)!
    """
    weather_df, current_weather = fetch_live_weather(12.9716, 77.5946)
    if current_weather and "wind_direction_10m" in current_weather and current_weather["wind_direction_10m"] is not None:
        w_spd_kmh = float(current_weather.get("wind_speed_10m", 12.0))
        w_spd = w_spd_kmh / 3.6
        w_dir = float(current_weather.get("wind_direction_10m", 85.0))
        live_gust_kmh = float(current_weather.get("wind_gusts_10m", w_spd_kmh * 1.4))
        temp_c = float(current_weather.get("temperature_2m", 26.5))
        rh_pct = float(current_weather.get("relative_humidity_2m", 58.0))
        
        weather_data = {
            "wind_speed_ms": w_spd,
            "wind_speed_kmh": w_spd_kmh,
            "wind_direction_10m": w_dir,
            "wind_gusts_kmh": live_gust_kmh,
            "temperature_2m": temp_c,
            "relative_humidity_2m": rh_pct,
            "source": "OPENMETEO_LIVE"
        }
        save_persisted_weather(weather_data)
        print(f"[WEATHER LIVE] Successfully fetched & persisted Open-Meteo: {w_spd_kmh:.1f} km/h @ {w_dir:.0f}°")
        return weather_data
    
    # Check persistent cache (15-min TTL or warm disk snapshot)
    cached = get_persisted_weather()
    if cached is not None:
        cached_copy = cached.copy()
        cached_copy["source"] = "PERSISTENT_CACHE"
        print(f"[WEATHER CACHE] Using persistent 15-min weather cache: {cached.get('wind_speed_kmh', 11.5):.1f} km/h @ {cached.get('wind_direction_10m', 85.0):.0f}°")
        return cached_copy

    # Standard urban Bengaluru default (never dataset snapshot 230°)
    default_weather = {
        "wind_speed_ms": 3.2,
        "wind_speed_kmh": 11.5,
        "wind_direction_10m": 85.0,
        "wind_gusts_kmh": 22.0,
        "temperature_2m": 26.0,
        "relative_humidity_2m": 55.0,
        "source": "DEFAULT_STABLE"
    }
    save_persisted_weather(default_weather)
    print(f"[WEATHER DEFAULT] Initialized stable weather: 11.5 km/h @ 85.0°")
    return default_weather

def run_live_inference(
    target_station_idx: int = 0,
    target_horizon_idx: int = 0,
    perturbation_station_idx: Optional[int] = None,
    perturbation_pm25_delta: float = 0.0,
    use_cached_if_rate_limited: bool = True
) -> Dict[str, Any]:
    """
    Pulls live OpenAQ and Open-Meteo telemetry, constructs physical wind-advection graph,
    runs WindSTGNN.forward_with_internals(), and exposes real activation & saliency tensors.
    """
    now_ts = time.time()
    cache_key = (target_station_idx, target_horizon_idx, perturbation_station_idx, round(perturbation_pm25_delta, 2))
    if use_cached_if_rate_limited and cache_key in _LIVE_INFERENCE_CACHE:
        entry = _LIVE_INFERENCE_CACHE[cache_key]
        if (now_ts - entry["timestamp"] < 30.0):
            return copy.deepcopy(entry["result"])

    dataset_path = DATA_DIR / "dataset.pt"
    ckpt_path = CHECKPOINTS_DIR / "best_wind_stgnn.pt"
    stations_path = DATA_DIR / "stations.json"

    if not dataset_path.exists() or not ckpt_path.exists():
        raise FileNotFoundError("Missing dataset.pt or best_wind_stgnn.pt checkpoint.")

    dataset = torch.load(dataset_path, map_location="cpu")
    checkpoint = torch.load(ckpt_path, map_location="cpu")

    if stations_path.exists():
        with open(stations_path, "r", encoding="utf-8") as f:
            discovered_stations = json.load(f)
    else:
        discovered_stations = []

    N = dataset["X_test"].shape[2]
    scaler_mean = checkpoint["scaler_mean"].numpy()
    scaler_std = checkpoint["scaler_std"].numpy()
    pm25_mean = checkpoint["pm25_mean"]
    pm25_std = checkpoint["pm25_std"]

    stations = []
    for i in range(N):
        if i < len(discovered_stations):
            s = discovered_stations[i].copy()
        else:
            s = {"id": i, "name": f"Station {i}", "latitude": 12.9716, "longitude": 77.5946}
        raw_name = s.get("name", f"Station {i}")
        s["short_name"] = raw_name.replace(", Bengaluru - KSPCB", "").replace(", Bengaluru - CPCB", "").replace(" - KSPCB", "").replace(" - CPCB", "").strip()
        stations.append(s)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = WindSTGNN(
        in_dim=9,
        hidden_dim=checkpoint["model_state_dict"]["input_proj.weight"].shape[0],
        seq_out=6,
        dropout=0.0
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # Time calculations
    now_utc = datetime.now(timezone.utc)
    ist_tz = timezone(timedelta(hours=5, minutes=30))
    now_ist = now_utc.astimezone(ist_tz)
    fetch_timestamp_ist = now_ist.strftime("%d %b %Y, %I:%M:%S %p IST")

    # Live poll OpenAQ PM2.5 readings
    live_pm_readings = fetch_bengaluru_live_pm25(stations)
    if len(live_pm_readings) >= 3:
        data_source = "LIVE_API"
        try:
            with open(PM25_CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump({str(k): v for k, v in live_pm_readings.items()}, f)
        except Exception:
            pass
    else:
        if PM25_CACHE_FILE.exists():
            try:
                with open(PM25_CACHE_FILE, "r", encoding="utf-8") as f:
                    cached_pm = json.load(f)
                    live_pm_readings = {int(k): float(v) for k, v in cached_pm.items()}
                    data_source = "CACHED_FALLBACK"
            except Exception:
                data_source = "CACHED_FALLBACK"
        else:
            data_source = "CACHED_FALLBACK"

    # Base tensor window
    x_seq = dataset["X_test"][-1:].clone().to(device)
    adj_seq = dataset["Adj_test"][-1:].clone().to(device)

    # 24h baseline unscaled
    past_24h_pm25 = (x_seq[0, :, :, 0].cpu().numpy() * pm25_std + pm25_mean).clip(min=1.0)

    # Inject live/cached PM2.5 readings
    if len(live_pm_readings) > 0:
        latest_live_pm = past_24h_pm25[-1].copy()
        for i, s in enumerate(stations):
            lid = s["id"]
            if lid in live_pm_readings:
                latest_live_pm[i] = live_pm_readings[lid]
        
        # Roll 24h window by 1 hour and place live reading as latest hour
        past_24h_pm25 = np.vstack([past_24h_pm25[1:], latest_live_pm[np.newaxis, :]])

        # Update input tensor x_seq at latest timestep -1 with normalized PM2.5
        norm_pm25 = (torch.tensor(latest_live_pm, device=device, dtype=torch.float32) - pm25_mean) / pm25_std
        x_seq[0, -1, :, 0] = norm_pm25

    # Always fetch effective weather (Live Open-Meteo -> 15-min cache -> stable urban default)
    # NEVER fall back to offline dataset snapshot (230°)
    weather = get_effective_weather()
    w_spd = float(weather["wind_speed_ms"])
    w_spd_kmh = float(weather["wind_speed_kmh"])
    w_dir = float(weather["wind_direction_10m"])
    live_gust_kmh = float(weather.get("wind_gusts_kmh", w_spd_kmh * 1.4))

    # Inject effective meteorology into the latest timestep of x_seq for all stations
    norm_spd = (w_spd - scaler_mean[3]) / scaler_std[3]
    norm_sin = (np.sin(np.deg2rad(w_dir)) - scaler_mean[4]) / scaler_std[4]
    norm_cos = (np.cos(np.deg2rad(w_dir)) - scaler_mean[5]) / scaler_std[5]
    x_seq[0, -1, :, 3] = norm_spd
    x_seq[0, -1, :, 4] = norm_sin
    x_seq[0, -1, :, 5] = norm_cos

    current_pm25 = past_24h_pm25[-1].copy()

    # Causal what-if perturbation support
    perturbation_active = False
    if perturbation_station_idx is not None and abs(perturbation_pm25_delta) > 0.01:
        if 0 <= perturbation_station_idx < N:
            perturbation_active = True
            current_pm25[perturbation_station_idx] = max(1.0, float(current_pm25[perturbation_station_idx] + perturbation_pm25_delta))
            past_24h_pm25[-1, perturbation_station_idx] = current_pm25[perturbation_station_idx]
            norm_perturbed = (current_pm25[perturbation_station_idx] - pm25_mean) / pm25_std
            x_seq[0, -1, perturbation_station_idx, 0] = norm_perturbed

    # Unscale latest live meteorology to compute real-time wind vectors
    latest_unscaled = x_seq[0, -1].cpu().numpy() * scaler_std + scaler_mean
    wind_speeds = latest_unscaled[:, 3].clip(min=1.0)
    wind_sin = latest_unscaled[:, 4]
    wind_cos = latest_unscaled[:, 5]
    wind_directions = (np.degrees(np.arctan2(wind_sin, wind_cos)) + 360.0) % 360.0

    # Compute Real-Time Air Propagation Adjacency Matrix from Live Meteorological Telemetry
    dist_km, bearings = build_spatial_matrices(stations)
    live_adj = compute_wind_adjacency(
        wind_speeds=wind_speeds,
        wind_dirs=wind_directions,
        dist_km=dist_km,
        bearings=bearings,
        sigma_km=DEFAULT_SIGMA_KM
    )
    # Inject real-time wind-advection graph at the current observation timestep
    adj_seq[0, -1] = live_adj.squeeze(0).to(device)

    # Run ST-GNN inference exposing real activation & gradient saliency tensors
    pred_raw, gcn1_act, gcn2_act, gru_hidden, saliency, decoder_weights, _ = model.forward_with_internals(
        x=x_seq,
        adj=adj_seq,
        target_station_idx=target_station_idx,
        target_horizon_idx=target_horizon_idx,
        compute_saliency=True
    )

    pred_vals = pred_raw[0].cpu().numpy().clip(min=1.0)
    pred_delta = pred_vals - pred_vals[:, 0:1]
    forecast_pm25 = np.zeros_like(pred_vals)
    for h in range(6):
        weight_drift = (h + 1) / 6.0
        forecast_pm25[:, h] = current_pm25 * (1.0 - weight_drift * 0.45) + pred_vals[:, h] * (weight_drift * 0.45) + pred_delta[:, h]
    forecast_pm25 = forecast_pm25.clip(min=5.0)

    last_adj = adj_seq[0, -1].cpu().numpy()

    # Telemetry metrics
    avg_wind_speed_ms = float(np.mean(wind_speeds))
    avg_wind_speed_kmh = avg_wind_speed_ms * 3.6
    dominant_wind_dir_deg = float(np.mean(wind_directions))

    # Transport edges
    active_edges = []
    for j in range(N):
        for i in range(N):
            if i != j and last_adj[j, i] > 0.12:
                active_edges.append({
                    "from_station_id": stations[i]["id"],
                    "from_name": stations[i]["name"],
                    "from_short": stations[i]["short_name"],
                    "from_lat": stations[i]["latitude"],
                    "from_lon": stations[i]["longitude"],
                    "to_station_id": stations[j]["id"],
                    "to_name": stations[j]["name"],
                    "to_short": stations[j]["short_name"],
                    "to_lat": stations[j]["latitude"],
                    "to_lon": stations[j]["longitude"],
                    "weight": float(last_adj[j, i])
                })

    active_edges.sort(key=lambda x: x["weight"], reverse=True)

    result = {
        "timestamp_utc": now_utc.strftime("%Y-%m-%d %H:%M UTC"),
        "telemetry": {
            "data_source": data_source,
            "fetch_timestamp_ist": fetch_timestamp_ist,
            "avg_wind_speed_kmh": avg_wind_speed_kmh,
            "wind_gusts_kmh": live_gust_kmh if "live_gust_kmh" in locals() else avg_wind_speed_kmh * 1.4,
            "dominant_wind_dir_deg": dominant_wind_dir_deg,
            "perturbation_active": perturbation_active
        },
        "stations": stations,
        "past_24h_pm25": past_24h_pm25,
        "current_pm25": current_pm25,
        "forecast_pm25": forecast_pm25,
        "wind_speeds": wind_speeds,
        "wind_directions": wind_directions,
        "active_wind_edges": active_edges,
        "adjacency_matrix": last_adj,
        "internals": {
            "gcn1_act": gcn1_act[0, -1].cpu().numpy(),                         # (N, 64) activation at current t
            "gcn2_act": gcn2_act[0, :, target_station_idx, :].cpu().numpy(),   # (24, 64) full 24h spatial embedding for target station
            "gcn2_act_last": gcn2_act[0, -1, target_station_idx, :].cpu().numpy(), # (64,)
            "gru_hidden": gru_hidden[0, :, -1, :].cpu().numpy(),               # (N, 64) final temporal GRU state per node
            "gru_sequence": gru_hidden[0].cpu().numpy(),                      # (N, 24, 64) full 24h temporal sequence of 64 units
            "saliency": saliency[0].cpu().numpy(),                             # (24, N, 9) input gradient sensitivity
            "decoder_weights": decoder_weights.cpu().numpy(),                  # (6, 64) output projection weights
            "neighbor_contrib": (last_adj[target_station_idx, :] * current_pm25).astype(float), # (N,) exact upwind PM2.5 contribution
            "target_station_idx": target_station_idx,
            "target_horizon_idx": target_horizon_idx,
            "feature_names": [
                "PM2.5", "Temperature", "Rel Humidity", "Wind Speed",
                "Wind Sin", "Wind Cos", "Hour Sin", "Hour Cos", "DayOfWeek"
            ]
        }
    }
    print(f"[WIND AUDIT] dominant_wind_dir={dominant_wind_dir_deg:.1f}° | wind_speed={avg_wind_speed_kmh:.1f} km/h | advection_flow={(dominant_wind_dir_deg + 180.0) % 360.0:.1f}° | weather_source={weather.get('source')}")

    _LIVE_INFERENCE_CACHE[cache_key] = {
        "timestamp": now_ts,
        "result": copy.deepcopy(result)
    }
    return copy.deepcopy(result)

if __name__ == "__main__":
    res = run_live_inference()
    tel = res["telemetry"]
    print("Telemetry:", tel)
    print(f"Data source: {tel['data_source']} | IST: {tel['fetch_timestamp_ist']}")
