import os
import time
import json
import logging
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import torch
from sklearn.preprocessing import StandardScaler
from dotenv import load_dotenv

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import (
    DATA_DIR, HISTORY_DAYS, SEQ_IN, SEQ_OUT,
    OPENAQ_BASE_URL, OPENMETEO_ARCHIVE_URL, DEFAULT_SIGMA_KM
)
from src.graph_physics import build_spatial_matrices, compute_wind_adjacency

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def get_openaq_headers():
    load_dotenv()
    api_key = os.getenv("OPENAQ_API_KEY")
    if not api_key:
        raise ValueError("OPENAQ_API_KEY not found in environment or .env file.")
    return {"X-API-Key": api_key.strip()}

def fetch_openaq_sensor_hours(
    sensor_id: int,
    dt_from: datetime,
    dt_to: datetime,
    max_retries: int = 4
) -> List[Dict]:
    """
    Fetches hourly sensor data between dt_from and dt_to using pagination and retry backoff.
    """
    headers = get_openaq_headers()
    iso_from = dt_from.strftime("%Y-%m-%dT%H:%M:%SZ")
    iso_to = dt_to.strftime("%Y-%m-%dT%H:%M:%SZ")
    
    all_records = []
    page = 1
    limit = 1000

    while True:
        url = (
            f"{OPENAQ_BASE_URL}/sensors/{sensor_id}/hours"
            f"?datetime_from={iso_from}&datetime_to={iso_to}&limit={limit}&page={page}"
        )
        
        success = False
        for attempt in range(max_retries):
            try:
                resp = requests.get(url, headers=headers, timeout=25)
                if resp.status_code == 200:
                    data = resp.json()
                    results = data.get("results", [])
                    all_records.extend(results)
                    success = True
                    if len(results) < limit:
                        # Reached the last page
                        return all_records
                    break
                elif resp.status_code in [429, 500, 502, 503, 504]:
                    sleep_sec = 2.0 * (attempt + 1)
                    logging.warning(f"Sensor {sensor_id} returned {resp.status_code}. Retrying in {sleep_sec}s...")
                    time.sleep(sleep_sec)
                else:
                    logging.warning(f"Sensor {sensor_id} failed with status {resp.status_code}: {resp.text[:100]}")
                    return all_records
            except Exception as e:
                sleep_sec = 2.0 * (attempt + 1)
                logging.warning(f"Sensor {sensor_id} request error ({e}). Retrying in {sleep_sec}s...")
                time.sleep(sleep_sec)

        if not success:
            logging.error(f"Failed to fetch sensor {sensor_id} page {page} after {max_retries} retries.")
            break
        
        page += 1
        time.sleep(0.15)  # Polite pause

    return all_records

def fetch_open_meteo_weather(
    lat: float,
    lon: float,
    start_date: str,
    end_date: str
) -> pd.DataFrame:
    """
    Fetches hourly weather variables (temperature, relative humidity, wind speed, wind direction)
    from Open-Meteo Historical Archive API.
    """
    url = (
        f"{OPENMETEO_ARCHIVE_URL}?latitude={lat:.4f}&longitude={lon:.4f}"
        f"&start_date={start_date}&end_date={end_date}"
        f"&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m"
    )
    for attempt in range(4):
        try:
            resp = requests.get(url, timeout=20)
            if resp.status_code == 200:
                data = resp.json()
                hourly = data.get("hourly", {})
                times = hourly.get("time", [])
                df = pd.DataFrame({
                    "timestamp": pd.to_datetime(times, utc=True),
                    "temperature": hourly.get("temperature_2m", []),
                    "relative_humidity": hourly.get("relative_humidity_2m", []),
                    "wind_speed": hourly.get("wind_speed_10m", []),
                    "wind_direction": hourly.get("wind_direction_10m", [])
                })
                df = df.set_index("timestamp").sort_index()
                return df
            else:
                logging.warning(f"Open-Meteo error {resp.status_code}. Retrying...")
                time.sleep(2.0 * (attempt + 1))
        except Exception as e:
            logging.warning(f"Open-Meteo request error: {e}. Retrying...")
            time.sleep(2.0 * (attempt + 1))

    raise RuntimeError(f"Failed to fetch Open-Meteo weather for ({lat}, {lon})")

def build_dataset():
    """
    Ingests 90 days of PM2.5, meteorological features, builds wind adjacency,
    constructs sliding windows, scales features, and saves data/dataset.pt.
    """
    stations_path = DATA_DIR / "stations.json"
    if not stations_path.exists():
        raise FileNotFoundError(f"Missing {stations_path}. Run discover_stations.py first.")

    with open(stations_path, "r", encoding="utf-8") as f:
        stations = json.load(f)

    N = len(stations)
    logging.info(f"Loaded {N} stations from {stations_path}")

    # Set up UTC hourly datetime index for 90 days up to yesterday
    now = datetime.now(timezone.utc)
    end_dt = (now - timedelta(days=1)).replace(minute=0, second=0, microsecond=0)
    start_dt = end_dt - timedelta(days=HISTORY_DAYS)

    hourly_index = pd.date_range(start=start_dt, end=end_dt, freq="1h", tz="UTC")
    total_hours = len(hourly_index)
    logging.info(f"Dataset target span: {start_dt.date()} to {end_dt.date()} ({total_hours} hourly steps)")

    start_date_str = start_dt.strftime("%Y-%m-%d")
    end_date_str = end_dt.strftime("%Y-%m-%d")

    # 1. Fetch PM2.5 data for each station
    pm25_series_dict = {}
    for idx, st in enumerate(stations):
        st_id = st["id"]
        s_id = st["pm25_sensor_id"]
        logging.info(f"[{idx+1}/{N}] Fetching PM2.5 for station '{st['name']}' (Sensor {s_id})...")
        records = fetch_openaq_sensor_hours(s_id, start_dt, end_dt)
        
        parsed_records = []
        for r in records:
            val = r.get("value")
            dt_raw = r.get("period", {}).get("datetimeFrom", {}).get("utc")
            if val is not None and dt_raw is not None and val >= 0:
                parsed_records.append({"timestamp": dt_raw, "pm25": float(val)})

        if parsed_records:
            df_st = pd.DataFrame(parsed_records)
            df_st["timestamp"] = pd.to_datetime(df_st["timestamp"], utc=True).dt.floor("1h")
            # Group by hourly floor to aggregate duplicates
            df_st = df_st.groupby("timestamp")["pm25"].mean()
            # Reindex to master hourly index
            df_st = df_st.reindex(hourly_index)
        else:
            df_st = pd.Series(index=hourly_index, data=np.nan)

        pm25_series_dict[st_id] = df_st

    pm25_df = pd.DataFrame(pm25_series_dict)  # Shape: (total_hours, N)

    # 2. Fetch Open-Meteo weather for each station
    temp_dict = {}
    rh_dict = {}
    ws_dict = {}
    wd_dict = {}

    for idx, st in enumerate(stations):
        st_id = st["id"]
        logging.info(f"[{idx+1}/{N}] Fetching weather for '{st['name']}' ({st['latitude']:.4f}, {st['longitude']:.4f})...")
        df_weather = fetch_open_meteo_weather(st["latitude"], st["longitude"], start_date_str, end_date_str)
        df_weather = df_weather.reindex(hourly_index).interpolate(method="time").bfill().ffill()

        temp_dict[st_id] = df_weather["temperature"]
        rh_dict[st_id] = df_weather["relative_humidity"]
        ws_dict[st_id] = df_weather["wind_speed"]
        wd_dict[st_id] = df_weather["wind_direction"]

    temp_df = pd.DataFrame(temp_dict)
    rh_df = pd.DataFrame(rh_dict)
    ws_df = pd.DataFrame(ws_dict)
    wd_df = pd.DataFrame(wd_dict)

    # 3. Clean and impute PM2.5 missing values:
    # Forward-fill gaps up to 3 hours; fill remaining missing values with spatial station median per timestamp
    logging.info("Imputing PM2.5 gaps (ffill up to 3h + spatial station median fallback)...")
    pm25_clean = pm25_df.ffill(limit=3)
    spatial_median = pm25_clean.median(axis=1)
    # Global fallback if spatial median is NaN for any time
    global_median = pm25_clean.stack().median()
    if pd.isna(global_median):
        global_median = 35.0  # Safe urban fallback
    spatial_median = spatial_median.fillna(global_median)

    for col in pm25_clean.columns:
        pm25_clean[col] = pm25_clean[col].fillna(spatial_median)

    # 4. Compute wind vectors (sin, cos) and temporal cyclical features (hour of day, day of week)
    # wd_df is in degrees [0, 360)
    wd_rad = np.deg2rad(wd_df.values)
    wind_sin = np.sin(wd_rad)
    wind_cos = np.cos(wd_rad)

    hours = hourly_index.hour.values[:, None]  # (T, 1)
    dayofweek = hourly_index.dayofweek.values[:, None]  # (T, 1)
    hour_sin = np.sin(2 * np.pi * hours / 24.0)
    hour_cos = np.cos(2 * np.pi * hours / 24.0)
    dow_sin = np.sin(2 * np.pi * dayofweek / 7.0)
    dow_cos = np.cos(2 * np.pi * dayofweek / 7.0)

    # Tile temporal features across N nodes: (T, N)
    hour_sin_nodes = np.tile(hour_sin, (1, N))
    hour_cos_nodes = np.tile(hour_cos, (1, N))
    dow_sin_nodes = np.tile(dow_sin, (1, N))
    dow_cos_nodes = np.tile(dow_cos, (1, N))

    # Feature stack: F = 9 features:
    # [pm25, temperature, relative_humidity, wind_speed, wind_sin, wind_cos, hour_sin, hour_cos, dow_sin]
    feature_list = [
        pm25_clean.values,      # 0: PM2.5 (target is future PM2.5)
        temp_df.values,        # 1: Temperature
        rh_df.values,          # 2: Relative Humidity
        ws_df.values,          # 3: Wind Speed
        wind_sin,              # 4: Wind sin
        wind_cos,              # 5: Wind cos
        hour_sin_nodes,        # 6: Hour sin
        hour_cos_nodes,        # 7: Hour cos
        dow_sin_nodes          # 8: Day of week sin
    ]
    # (T, N, F)
    raw_node_features = np.stack(feature_list, axis=-1).astype(np.float32)
    T, _, F = raw_node_features.shape
    logging.info(f"Raw node feature tensor shape: {raw_node_features.shape} (T={T}, N={N}, F={F})")

    # 5. Compute full Adjacency trajectory Adj: shape (T, N, N)
    logging.info("Building wind-advection adjacency matrices for all timesteps...")
    dist_km, bearings = build_spatial_matrices(stations)
    ws_tensor = torch.from_numpy(ws_df.values).float()
    wd_tensor = torch.from_numpy(wd_df.values).float()

    adj_tensor = compute_wind_adjacency(
        wind_speeds=ws_tensor,
        wind_dirs=wd_tensor,
        dist_km=dist_km,
        bearings=bearings,
        sigma_km=DEFAULT_SIGMA_KM
    )  # shape: (T, N, N)
    adj_np = adj_tensor.numpy()

    # 6. Build sliding windows (SEQ_IN=24, SEQ_OUT=6)
    # Input X: (SEQ_IN, N, F)
    # Input Adj: (SEQ_IN, N, N)
    # Target Y: (N, SEQ_OUT) -> PM2.5 values across next 6 hours
    num_samples = T - SEQ_IN - SEQ_OUT + 1
    if num_samples <= 0:
        raise ValueError(f"Not enough timesteps ({T}) for SEQ_IN={SEQ_IN} and SEQ_OUT={SEQ_OUT}")

    X_list = []
    Adj_list = []
    Y_list = []

    for i in range(num_samples):
        x_win = raw_node_features[i : i + SEQ_IN]                       # (24, N, F)
        adj_win = adj_np[i : i + SEQ_IN]                               # (24, N, N)
        # Target Y is PM2.5 at future timesteps [i + SEQ_IN : i + SEQ_IN + SEQ_OUT]
        # pm25_clean is (T, N). Target shape required: (N, 6)
        y_win = pm25_clean.values[i + SEQ_IN : i + SEQ_IN + SEQ_OUT].T  # (N, 6)

        X_list.append(x_win)
        Adj_list.append(adj_win)
        Y_list.append(y_win)

    X_arr = np.array(X_list, dtype=np.float32)       # (Samples, 24, N, F)
    Adj_arr = np.array(Adj_list, dtype=np.float32)   # (Samples, 24, N, N)
    Y_arr = np.array(Y_list, dtype=np.float32)       # (Samples, N, 6)

    logging.info(f"Constructed {num_samples} sliding windows. Shapes: X={X_arr.shape}, Adj={Adj_arr.shape}, Y={Y_arr.shape}")

    # 7. Chronological Split (70% train, 15% val, 15% test)
    n_train = int(num_samples * 0.70)
    n_val = int(num_samples * 0.15)
    n_test = num_samples - n_train - n_val

    X_train_raw = X_arr[:n_train]
    Adj_train = Adj_arr[:n_train]
    Y_train = Y_arr[:n_train]

    X_val_raw = X_arr[n_train : n_train + n_val]
    Adj_val = Adj_arr[n_train : n_train + n_val]
    Y_val = Y_arr[n_train : n_train + n_val]

    X_test_raw = X_arr[n_train + n_val :]
    Adj_test = Adj_arr[n_train + n_val :]
    Y_test = Y_arr[n_train + n_val :]

    # 8. Fit StandardScaler on Training set (flatten across samples, time, nodes)
    scaler = StandardScaler()
    # Flatten X_train_raw to (-1, F)
    X_train_flat = X_train_raw.reshape(-1, F)
    scaler.fit(X_train_flat)

    # Transform all splits
    X_train = scaler.transform(X_train_flat).reshape(X_train_raw.shape).astype(np.float32)
    X_val = scaler.transform(X_val_raw.reshape(-1, F)).reshape(X_val_raw.shape).astype(np.float32)
    X_test = scaler.transform(X_test_raw.reshape(-1, F)).reshape(X_test_raw.shape).astype(np.float32)

    # Compute PM2.5 specific scalar mean and std (feature index 0) for loss / inverse scaling
    pm25_mean = float(scaler.mean_[0])
    pm25_std = float(scaler.scale_[0])

    dataset = {
        "X_train": torch.from_numpy(X_train),
        "Adj_train": torch.from_numpy(Adj_train),
        "Y_train": torch.from_numpy(Y_train),
        "X_val": torch.from_numpy(X_val),
        "Adj_val": torch.from_numpy(Adj_val),
        "Y_val": torch.from_numpy(Y_val),
        "X_test": torch.from_numpy(X_test),
        "Adj_test": torch.from_numpy(Adj_test),
        "Y_test": torch.from_numpy(Y_test),
        "scaler_mean": torch.from_numpy(scaler.mean_.astype(np.float32)),
        "scaler_std": torch.from_numpy(scaler.scale_.astype(np.float32)),
        "pm25_mean": pm25_mean,
        "pm25_std": pm25_std,
        "feature_names": [
            "pm25", "temperature", "relative_humidity", "wind_speed",
            "wind_sin", "wind_cos", "hour_sin", "hour_cos", "dow_sin"
        ],
        "station_ids": [s["id"] for s in stations],
        "dist_km": torch.from_numpy(dist_km),
        "bearings": torch.from_numpy(bearings)
    }

    out_dataset_path = DATA_DIR / "dataset.pt"
    torch.save(dataset, out_dataset_path)
    logging.info(f"Successfully saved clean dataset to {out_dataset_path}")
    logging.info(f"Train samples: {len(X_train)}, Val samples: {len(X_val)}, Test samples: {len(X_test)}")

    return dataset

if __name__ == "__main__":
    build_dataset()
