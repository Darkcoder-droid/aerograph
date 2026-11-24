# AeroGraph

Spatio-Temporal Graph Neural Network for Atmospheric Dispersion and Urban Air Quality Forecasting

AeroGraph is a physics-informed spatio-temporal graph neural network (WindSTGNN) designed for multi-horizon urban PM2.5 forecasting. Rather than treating atmospheric monitoring stations as independent time series, AeroGraph continuously reconstructs a dynamic spatial graph guided by real-time meteorological wind velocity vectors.

The system streams live ground-truth telemetry from continuous ambient air quality monitoring stations (CAAQMS) via OpenAQ v3, coupled with local atmospheric dynamics from Open-Meteo, to generate 1-hour to 6-hour predictive horizons with full spatial advection modeling.

---

## Architecture Overview

Urban air pollution is an advection-diffusion process governed by atmospheric fluid dynamics. AeroGraph models this process by coupling a dynamic graph convolution layer with a recurrent temporal lattice.

### 1. Dynamic Physical Advection Adjacency Tensor

Traditional graph neural networks rely on static, distance-only spatial graphs. AeroGraph evaluates dynamic directed coupling between station pairs $(i, j)$ at each time step $t$:

$$A_{ij}(t) = \exp\left(-\frac{d_{ij}}{\sigma}\right) \cdot \max\left(0, \cos(\theta_{\text{wind}}(t) - \theta_{ij})\right)$$

Where:
- $d_{ij}$ is the pairwise geodesic distance computed via the Haversine formula.
- $\sigma$ is a spatial decay kernel bandwidth (default: 10 km).
- $\theta_{\text{wind}}(t)$ is the dominant meteorological wind direction at time $t$.
- $\theta_{ij}$ is the directional azimuth bearing from station $i$ to station $j$.
- Self-loops $A_{ii}(t) = 1.0$ ensure station-level autoregressive memory retention.

The resulting matrix is degree-normalized via symmetric Laplacian scaling:

$$\tilde{A}(t) = \tilde{D}^{-\frac{1}{2}} (A(t) + I) \tilde{D}^{-\frac{1}{2}}$$

### 2. Spatio-Temporal Convolution (WindSTGNN)

The network processes a 24-hour lookback window across $N$ stations with 9 telemetry feature channels:
- Observed PM2.5 ($\mu\text{g/m}^3$)
- Temperature ($^\circ\text{C}$) and Relative Humidity (%)
- Wind speed ($\text{km/h}$)
- Cyclic wind direction encoding: $\sin(\theta_{\text{wind}}), \cos(\theta_{\text{wind}})$
- Cyclic diurnal temporal encoding: $\sin(2\pi t / 24), \cos(2\pi t / 24)$

The forward pass consists of:
1. **Spatial Aggregation**: A spatial graph convolution layer aggregates incoming upstream pollutant plumes:
   $$Z_t = \text{ReLU}\left(\tilde{A}(t) X_t W_s + b_s\right)$$
2. **Temporal Dynamics**: A 64-channel Gated Recurrent Unit (GRU) processes the spatial representations sequentially over the 24-hour lookback horizon to maintain temporal momentum:
   $$h_t = \text{GRU}(Z_t, h_{t-1})$$
3. **Multi-Horizon Synaptic Decoder**: An affine projection network decodes the latent hidden state $h_{24} \in \mathbb{R}^{N \times 64}$ into trajectories across 6 forward hourly steps:
   $$\hat{Y} = h_{24} W_{\text{out}} + b_{\text{out}} \quad \in \mathbb{R}^{N \times 6}$$

---

## Validation & Benchmarks

The model was evaluated against test set observations across the Bengaluru CAAQMS network (90-day baseline dataset, 80/10/10 train/val/test split).

| Model Architecture | MAE (µg/m³) | RMSE (µg/m³) | R² Score | Advection Mechanism |
| :--- | :---: | :---: | :---: | :--- |
| **AeroGraph (WindSTGNN)** | **7.82** | **11.45** | **0.879** | Dynamic Wind-Guided $A(t)$ |
| Temporal-Only GRU (No Graph) | 12.65 | 17.80 | 0.684 | None (Independent Nodes) |
| Static GCN + GRU | 10.40 | 14.92 | 0.771 | Static Haversine Kernel |
| Naive Persistence ($y_{t+h} = y_t$) | 18.42 | 25.10 | 0.421 | None |

AeroGraph achieves a **38.2% MAE improvement** over single-station temporal models by explicitly tracking incoming upwind plumes before they reach target sensors.

---

## System Structure

```
aerograph/
├── config.py                 # Network coordinates, hyperparameters, and city bounds
├── server.py                 # FastAPI backend with ASGI lifespan cache management
├── app.py                    # Streamlit analytical exploration workbench
├── requirements.txt          # Python dependencies
├── .env.example              # Environment variables template
├── data/
│   ├── stations.json         # CAAQMS station coordinates and operational metadata
│   └── bengaluru_real_osm.json # Geospatial boundaries and network topology
├── frontend/
│   ├── index.html            # Single-page interface (MapLibre 3D, WebGL particle flow)
│   └── aerograph_logo.svg    # Official AeroGraph vector brand mark
└── src/
    ├── model.py              # Pure-PyTorch WindSTGNN architecture (MPS / CUDA compatible)
    ├── graph_physics.py      # Haversine distance and dynamic advection tensor builders
    ├── live_stream.py        # Real-time OpenAQ v3 and Open-Meteo polling pipeline
    ├── discover_stations.py  # Utility for querying active OpenAQ CAAQMS sites
    ├── ingest_history.py     # Historical telemetry extraction pipeline
    └── train.py              # Training loop with validation and checkpoint management
```

---

## Key Features

- **Live Multi-Station Mesh**: Tracks 11 continuous monitoring stations across industrial, commercial, and residential urban zones.
- **WebGL Atmospheric Streamlines**: Interactive Eulerian particle flow simulation driven by live local wind speed and azimuth bearing.
- **Interactive Map Compass**: Hardware-accelerated compass rose aligned synchronously to true geographic North with single-click orientation reset.
- **Neural Studio Synapse Inspector**: Visualizes the 8x8 GRU hidden state lattice ($h_t \in \mathbb{R}^{64}$) and individual synaptic fan-out paths to target forecast horizons.
- **Counterfactual Emission Simulation**: Interactive testing of sudden +80 µg/m³ industrial emission spikes to evaluate downwind plume propagation.
- **No Heavy C++ Graph Dependencies**: Implemented in pure PyTorch using tensor operations and Einstein summation (`torch.einsum`), guaranteeing cross-platform compatibility across Apple Silicon (Metal Performance Shaders), CUDA, and CPU.

---

## Getting Started

### Prerequisites

- Python 3.10+
- An OpenAQ API key (optional for basic usage; public endpoints are supported)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Darkcoder-droid/aerograph.git
   cd aerograph
   ```

2. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Configure environment variables (optional):
   ```bash
   cp .env.example .env
   # Edit .env to add your OPENAQ_API_KEY if available
   ```

### Running the Services

#### Web Application (FastAPI + WebGL Interface)
```bash
python3 -m uvicorn server:app --host 127.0.0.1 --port 8000 --reload
```
Open `http://127.0.0.1:8000` in any modern browser to view the live 3D operations dashboard and neural lattice.

#### Exploratory Analytics Dashboard (Streamlit)
```bash
streamlit run app.py
```

---

## API Reference

The backend provides REST endpoints for automated telemetry access:

### `GET /api/telemetry`
Returns the complete model state, observed sensor data, dynamic adjacency matrix, and forecasted trajectories.

Parameters:
- `target_station_idx` (int, default: 0): Selected target station index (0 to 10).
- `horizon_idx` (int, default: 0): Forecast horizon index (0 = +1h, 5 = +6h).

### `POST /api/poll`
Forces an immediate cache invalidation and triggers live OpenAQ and Open-Meteo polling followed by a model forward pass.

---

## Technical Specifications

- **Target City**: Bengaluru, Karnataka, India (`12.9716° N, 77.5946° E`, 25 km bounding radius)
- **Monitoring Network**: 11 Continuous Ambient Air Quality Monitoring Stations (CPCB / KSPCB)
- **Temporal Resolution**: 1-hour IST intervals with 15-minute polling cadence
- **Hardware Acceleration**: Apple Silicon MPS (`torch.device("mps")`) or NVIDIA CUDA (`torch.device("cuda")`)

---

## License

This project is licensed under the Apache 2.0 License. See the LICENSE file for details.
