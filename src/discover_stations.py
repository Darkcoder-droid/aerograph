import os
import json
import logging
from datetime import datetime, timezone, timedelta
import requests
from dotenv import load_dotenv

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import OPENAQ_BASE_URL, DATA_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Bengaluru Full Metropolitan Bounding Box
BENGALURU_BBOX = "77.45,12.80,77.78,13.15"

def get_openaq_headers():
    load_dotenv()
    api_key = os.getenv("OPENAQ_API_KEY")
    if not api_key:
        raise ValueError("OPENAQ_API_KEY not found in environment or .env file.")
    return {"X-API-Key": api_key.strip()}

def clean_station_name(raw_name: str) -> str:
    cleaned = (
        raw_name.replace(", Bengaluru - KSPCB", "")
        .replace(", Bengaluru - CPCB", "")
        .replace(" - KSPCB", "")
        .replace(" - CPCB", "")
        .replace("blore_India_07_14_official", "Indiranagar")
        .replace("RVCE-Mailasandra", "Kengeri / RVCE")
        .strip()
    )
    return cleaned

def discover_stations(max_stations: int = 20):
    """
    Queries OpenAQ v3 for monitoring stations across Bengaluru's full metropolitan bounding box.
    Filters for stations that have active pm25 sensors updated within the last 7 days.
    """
    headers = get_openaq_headers()
    url = f"{OPENAQ_BASE_URL}/locations?bbox={BENGALURU_BBOX}&limit=100"
    
    logging.info(f"Querying OpenAQ v3 locations in bbox={BENGALURU_BBOX}...")
    response = requests.get(url, headers=headers, timeout=20)
    if response.status_code != 200:
        raise RuntimeError(f"OpenAQ API error {response.status_code}: {response.text}")

    data = response.json()
    results = data.get("results", [])
    logging.info(f"Fetched {len(results)} candidate locations across metropolitan Bengaluru.")

    now = datetime.now(timezone.utc)
    active_cutoff = now - timedelta(days=7)

    valid_stations = []

    for loc in results:
        coords = loc.get("coordinates")
        if not coords or coords.get("latitude") is None or coords.get("longitude") is None:
            continue

        dt_last_info = loc.get("datetimeLast")
        if not dt_last_info or not dt_last_info.get("utc"):
            continue

        try:
            dt_last = datetime.fromisoformat(dt_last_info["utc"].replace("Z", "+00:00"))
        except Exception:
            continue

        if dt_last < active_cutoff:
            continue

        # Extract sensor mappings
        sensors_raw = loc.get("sensors", [])
        sensor_map = {}
        for s in sensors_raw:
            param_name = s.get("parameter", {}).get("name", "").lower()
            s_id = s.get("id")
            if param_name in ["pm25", "pm10", "no2"]:
                if param_name not in sensor_map:
                    sensor_map[param_name] = []
                sensor_map[param_name].append(s_id)

        if "pm25" not in sensor_map or not sensor_map["pm25"]:
            continue

        pm25_sensor_id = max(sensor_map["pm25"])
        pm10_sensor_id = max(sensor_map["pm10"]) if "pm10" in sensor_map else None
        no2_sensor_id = max(sensor_map["no2"]) if "no2" in sensor_map else None

        short = clean_station_name(loc.get("name", "Station"))

        station_dict = {
            "id": loc.get("id"),
            "name": loc.get("name"),
            "short_name": short,
            "latitude": float(coords["latitude"]),
            "longitude": float(coords["longitude"]),
            "datetime_last": dt_last_info["utc"],
            "pm25_sensor_id": pm25_sensor_id,
            "pm10_sensor_id": pm10_sensor_id,
            "no2_sensor_id": no2_sensor_id,
            "all_pm25_sensor_ids": sensor_map["pm25"]
        }
        valid_stations.append(station_dict)

    logging.info(f"Identified {len(valid_stations)} active stations in metropolitan bbox.")
    valid_stations.sort(key=lambda x: x["datetime_last"], reverse=True)
    selected_stations = valid_stations[:max_stations]

    out_path = DATA_DIR / "stations.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(selected_stations, f, indent=2)

    logging.info(f"Saved {len(selected_stations)} stations to {out_path}")
    return selected_stations

if __name__ == "__main__":
    discover_stations()
