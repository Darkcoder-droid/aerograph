# `AGENT_SPEC.md`: Wind-Informed Spatio-Temporal GNN Air Quality Forecaster

## 1. System Context & Hardware Constraints
* **Target Machine:** Apple MacBook Pro M1 Pro (32GB Unified Memory, macOS).
* **Acceleration:** Use PyTorch native Apple Silicon backend (`torch.device("mps")` if `torch.backends.mps.is_available()` else `"cpu"`).
* **Strict Dependency Rule:** Do **NOT** install `torch-geometric`, `dgl`, or C++-compiled graph libraries (they cause Apple Silicon wheel/build failures). Implement all graph convolutions in **pure PyTorch** via `torch.einsum` and dense adjacency tensors.
* **Target Domain:** Urban Air Quality & Pollution Plume Forecasting (Default City: **Bengaluru, India** — `lat: 12.9716, lon: 77.5946`, radius: `25000`m).

---

## 2. Repository Structure (Target State)
The agent must scaffold and maintain this exact directory layout:

```text
air-plume-gnn/
├── AGENT_SPEC.md
├── .env.example              # OPENAQ_API_KEY=your_key_here
├── requirements.txt
├── config.py                 # City coordinates, hyperparameters, paths
├── data/
│   ├── stations.json         # Validated OpenAQ station metadata & coordinates
│   └── dataset.pt            # Normalized tensors: X, Adj, Y, scaler_stats
├── src/
│   ├── discover_stations.py  # Queries OpenAQ v3 for active stations
│   ├── ingest_history.py     # Fetches 90 days OpenAQ v3 + Open-Meteo archive
│   ├── graph_physics.py      # Haversine distance + wind-advection adjacency builder
│   ├── model.py              # Pure-PyTorch WindSTGNN (Dynamic GCN + GRU)
│   ├── train.py              # MPS training loop, validation, checkpointing
│   └── live_stream.py        # Real-time OpenAQ v3 + Open-Meteo poller & inference
├── checkpoints/
│   └── best_wind_stgnn.pt    # Saved weights + normalization stats
└── app.py                    # Streamlit + PyDeck 3D interactive plume dashboard
```

---

## 3. Implementation Phases

### Phase 1: Environment Scaffolding & Live API Discovery

* **Objective:** Initialize project structure, verify dependencies, and discover active OpenAQ v3 monitoring stations in Bengaluru.
* **Tasks:**
  * Create `requirements.txt` with `torch`, `numpy`, `pandas`, `requests`, `python-dotenv`, `scikit-learn`, `streamlit`, `pydeck`, and `plotly`.
  * Create `config.py` storing `CITY_CENTER = (12.9716, 77.5946)`, `RADIUS_M = 25000`, `HISTORY_DAYS = 90`, `SEQ_IN = 24`, `SEQ_OUT = 6`, and `OPENAQ_BASE_URL = "https://api.openaq.org/v3"`.
  * Write `src/discover_stations.py` to query `GET /v3/locations?coordinates=12.9716,77.5946&radius=25000&limit=50` with header `X-API-Key: $OPENAQ_API_KEY`.
  * Filter for stations that have active `pm25` sensors (and optionally `pm10`, `no2`) updated within the last 7 days, saving the top 10–15 stations (IDs, names, sensor IDs, lat/lon) to `data/stations.json`.
* **Verification:** `data/stations.json` exists and contains at least 8 valid stations with non-null coordinates and `pm25` sensor IDs.

---

### Phase 2: Historical Data Ingestion & Wind-Graph Tensor Construction

* **Objective:** Build a clean, aligned hourly spatio-temporal dataset ($X$, $\text{Adj}$, $Y$) combining air quality readings and meteorological wind vectors.
* **Tasks:**
  * **Implement `src/graph_physics.py`:**
    * Compute pairwise Haversine distances $d_{ij}$ (km) and forward bearings $\theta_{i \to j}$ between all stations in `data/stations.json`.
    * Implement `compute_wind_adjacency(wind_speeds, wind_dirs, sigma_km=8.0) -> torch.Tensor` of shape $(N, N)$. Meteorological `wind_direction_10m` denotes where wind blows from, so flow direction is $(\theta_{\text{wind}} + 180^\circ) \pmod{360^\circ}$. Weight directed edges $j \leftarrow i$ by:
      $$w_{ji} = \exp\left(-\frac{d_{ij}^2}{\sigma^2}\right) \cdot \max\left(0, \cos(\theta_{\text{flow}, i} - \theta_{i \to j})\right) \cdot \left(1 + \min\left(\frac{v_i}{10}, 1.5\right)\right)$$
    * Add self-loops (`np.eye(N)`) and row-normalize each matrix.
  * **Implement `src/ingest_history.py`:**
    * For each station's `pm25` (and available `pm10`, `no2`) sensor IDs, fetch the past 90 days of hourly data from `GET /v3/sensors/{sensors_id}/hours` (chunking by `datetime_from` and `datetime_to` with polite backoff to avoid timeouts).
    * Fetch aligned 90-day hourly weather (`temperature_2m`, `relative_humidity_2m`, `wind_speed_10m`, `wind_direction_10m`) for all station coordinates via [Open-Meteo Historical Weather API](https://archive-api.open-meteo.com/v1/archive).
    * Resample to a strict hourly UTC index. Forward-fill gaps up to 3 hours; fill remaining missing sensor values with spatial station medians.
    * Build sliding windows of length `SEQ_IN = 24` to predict PM2.5 for the next `SEQ_OUT = 6` hours across all $N$ nodes. Split chronologically (70% train, 15% val, 15% test) and save `data/dataset.pt` with `StandardScaler` mean and std.
* **Verification:** `data/dataset.pt` loads cleanly; `X_train` has shape `(Samples, 24, N, F)`, `Adj_train` has shape `(Samples, 24, N, N)` with zero NaNs, and `Y_train` has shape `(Samples, N, 6)`.

---

### Phase 3: Pure-PyTorch WindSTGNN Model & MPS Training

* **Objective:** Implement and train the Spatio-Temporal Graph Neural Network on Apple Silicon `mps`.
* **Tasks:**
  * **Implement `src/model.py`:**
    * `DynamicGraphConv(in_dim, out_dim)`: applies linear projection followed by `torch.einsum("btij,btjf->btif", adj, support)` and GELU.
    * `WindSTGNN`: 2 residual `DynamicGraphConv` layers (`hidden_dim=64`) $\to$ reshape to `(B*N, T, 64)` $\to$ 2-layer `nn.GRU` (`dropout=0.1`) $\to$ MLP prediction head outputting `(B, N, 6)` future PM2.5 values.
  * **Implement `src/train.py`:**
    * Train on `device="mps"` using `HuberLoss(delta=10.0)`, `AdamW(lr=1e-3, weight_decay=1e-4)`, and `ReduceLROnPlateau` for 40 epochs (`batch_size=32`).
    * Log Train/Val MAE and RMSE (in real $\mu\text{g/m}^3$ units) per epoch, plus compare against a persistence baseline (`pred = last_observed_pm25`).
    * Save best checkpoint to `checkpoints/best_wind_stgnn.pt`.
* **Verification:** Training completes in under 15 minutes on M1 Pro with zero runtime errors, and validation MAE beats the persistence baseline on the 3-hour and 6-hour horizons.

---

### Phase 4: Real-Time Stream Poller & Inference Engine

* **Objective:** Build a resilient live-data fetcher that pulls the latest 24 hours of live sensor and wind data and outputs 6-hour forecasts plus active GNN wind-advection arcs.
* **Tasks:**
  * **Implement `src/live_stream.py`:**
    * Fetch the last 24 hours of live hourly measurements from OpenAQ v3 (`GET /v3/sensors/{sensors_id}/hours`) and live weather/wind vectors from [Open-Meteo Forecast API](https://api.open-meteo.com/v1/forecast?past_days=1&forecast_days=1).
    * Include a graceful fallback mode (`use_cached_if_rate_limited=True`) using the latest test window so demos never crash if OpenAQ throttles requests.
    * Run `WindSTGNN` in `torch.no_grad()` mode to return `current_pm25` $(N,)$, `forecast_pm25` $(N, 6)$, and `active_wind_edges` where $\text{adj}[-1, j, i] > 0.12$.
* **Verification:** Running `python src/live_stream.py` prints a formatted table of Bengaluru stations, live PM2.5, +3h/+6h predictions, and top upwind-to-downwind transport corridors in under 5 seconds.

---

### Phase 5: Interactive 3D Plume Dashboard (`app.py`)

* **Objective:** Build a dark-mode Streamlit + PyDeck geospatial control room that visualizes live pollution columns, GNN wind-transport arcs, and 6-hour forecast animations.
* **Tasks:**
  * **Implement `app.py` with:**
    * **Top KPI Strip:** City-wide average PM2.5, worst-affected neighborhood, peak predicted 6h hotspot, and WHO/Indian CPCB Air Quality health advisory badge.
    * **Interactive Forecast Horizon Slider:** Now (Live), +1h, +2h, +3h, +4h, +5h, +6h.
    * **PyDeck 3D Map (`pdk.Deck`):**
      * `ColumnLayer`: Extrudes each station in 3D proportional to PM2.5 at the selected slider horizon, color-coded from green ($<30$) $\to$ yellow ($30\text{–}60$) $\to$ orange ($60\text{–}120$) $\to$ crimson ($>120$).
      * `ArcLayer`: Renders animated directional arcs between stations weighted by the GNN's live wind-advection matrix $A_t$, visually demonstrating spatial pollution propagation.
    * **Station Detail Drawer:** Selecting a station renders a Plotly time-series chart showing the past 24h ground truth + 6h predicted trajectory + local wind speed/direction badge.
    * **Auto-Refresh Toggle:** Optional 5-minute live polling refresh loop.
* **Verification:** `streamlit run app.py` launches cleanly without console errors; moving the +1h to +6h slider dynamically updates column heights, colors, and health advisories.
