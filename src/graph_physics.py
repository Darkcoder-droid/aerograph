import math
import torch
import numpy as np
from typing import List, Dict, Tuple, Union

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Computes great-circle distance between two GPS points in kilometers.
    """
    R = 6371.0  # Earth radius in km
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (math.sin(delta_phi / 2.0) ** 2 +
         math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2)
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return R * c

def forward_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Computes initial bearing (forward azimuth) from point 1 to point 2 in degrees [0, 360).
    """
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_lambda = math.radians(lon2 - lon1)

    y = math.sin(delta_lambda) * math.cos(phi2)
    x = (math.cos(phi1) * math.sin(phi2) -
         math.sin(phi1) * math.cos(phi2) * math.cos(delta_lambda))
    initial_bearing = math.atan2(y, x)
    # Convert from radians to degrees and normalize to [0, 360)
    return (math.degrees(initial_bearing) + 360.0) % 360.0

def build_spatial_matrices(stations: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given a list of station dictionaries with 'latitude' and 'longitude',
    returns:
      - dist_km: (N, N) pairwise distance in km
      - bearings: (N, N) forward bearing theta_{i -> j} in degrees
    """
    N = len(stations)
    dist_km = np.zeros((N, N), dtype=np.float32)
    bearings = np.zeros((N, N), dtype=np.float32)

    for i in range(N):
        for j in range(N):
            if i == j:
                dist_km[i, j] = 0.0
                bearings[i, j] = 0.0
            else:
                lat_i, lon_i = stations[i]["latitude"], stations[i]["longitude"]
                lat_j, lon_j = stations[j]["latitude"], stations[j]["longitude"]
                dist_km[i, j] = haversine_distance(lat_i, lon_i, lat_j, lon_j)
                bearings[i, j] = forward_bearing(lat_i, lon_i, lat_j, lon_j)

    return dist_km, bearings

def compute_wind_adjacency(
    wind_speeds: Union[np.ndarray, torch.Tensor],
    wind_dirs: Union[np.ndarray, torch.Tensor],
    dist_km: Union[np.ndarray, torch.Tensor],
    bearings: Union[np.ndarray, torch.Tensor],
    sigma_km: float = 8.0
) -> torch.Tensor:
    """
    Constructs wind-advection adjacency matrix A_t of shape (N, N) or (T, N, N).
    
    Advection formula from AGENT_SPEC.md:
      Meteorological wind_direction_10m denotes where wind blows from.
      Flow direction: theta_{flow, i} = (wind_dir_i + 180) % 360.
      Directed edge weight j <- i (pollution blowing from station i to station j):
      w_{ji} = exp(-d_{ij}^2 / sigma^2) * max(0, cos(theta_{flow, i} - theta_{i -> j})) * (1 + min(v_i / 10, 1.5))
      
      Self-loops added (np.eye(N) or I) and row-normalized (sum across columns = 1).

    Parameters:
      wind_speeds: (N,) or (T, N) wind speeds in m/s or km/h
      wind_dirs: (N,) or (T, N) wind directions in degrees [0, 360)
      dist_km: (N, N) pairwise distance in km
      bearings: (N, N) theta_{i -> j} bearing in degrees
      sigma_km: spatial decay scale (default 8.0 km)

    Returns:
      adj: torch.Tensor of shape (N, N) or (T, N, N), row-normalized with self-loops.
    """
    if isinstance(wind_speeds, np.ndarray):
        wind_speeds = torch.from_numpy(wind_speeds).float()
    if isinstance(wind_dirs, np.ndarray):
        wind_dirs = torch.from_numpy(wind_dirs).float()
    if isinstance(dist_km, np.ndarray):
        dist_km = torch.from_numpy(dist_km).float()
    if isinstance(bearings, np.ndarray):
        bearings = torch.from_numpy(bearings).float()

    is_batched = (wind_speeds.ndim == 2)
    if not is_batched:
        # Expand to (1, N) for unified computation
        wind_speeds = wind_speeds.unsqueeze(0)
        wind_dirs = wind_dirs.unsqueeze(0)

    T, N = wind_speeds.shape
    device = wind_speeds.device

    # dist_km is (N, N), expand to (1, N, N)
    d_sq = (dist_km ** 2).unsqueeze(0)  # (1, N, N)
    spatial_weight = torch.exp(-d_sq / (sigma_km ** 2))  # (1, N, N)

    # Flow direction: where wind is blowing TO
    # theta_flow is (T, N) -> expand to (T, N, 1) representing source node i
    flow_dir = (wind_dirs + 180.0) % 360.0
    flow_dir_rad = torch.deg2rad(flow_dir).unsqueeze(-1)  # (T, N, 1)

    # bearings is theta_{i -> j} of shape (N, N) -> expand to (1, N, N)
    bearing_rad = torch.deg2rad(bearings).unsqueeze(0)  # (1, N, N)

    # Alignment angle: cos(theta_{flow, i} - theta_{i -> j})
    angle_diff = flow_dir_rad - bearing_rad  # (T, N, N)
    directional_align = torch.clamp(torch.cos(angle_diff), min=0.0)  # (T, N, N)

    # Velocity scaling: 1 + min(v_i / 10, 1.5)
    # v_i is source node speed (T, N) -> expand to (T, N, 1)
    velocity_scale = 1.0 + torch.clamp(wind_speeds / 10.0, max=1.5).unsqueeze(-1)  # (T, N, 1)

    # Edge weight w_{ji} represents flow into node j from node i.
    # The source is i, target is j:
    # flow_dir and velocity are from source i (dim 1).
    # directional_align[t, i, j] is cos(flow_i - bearing_{i -> j}).
    # spatial_weight[0, i, j] is distance between i and j.
    # Therefore, transport from i to j is:
    # W_{ij} = transport from i to j.
    # In graph convention for message passing j <- i (adj[j, i]):
    # w_{ji} is edge from i to j.
    w_source_to_target = spatial_weight * directional_align * velocity_scale  # (T, N, N): [t, i, j]

    # Transpose so dim -2 is target j, dim -1 is source i -> adj[t, j, i] = flow from i to j
    adj = w_source_to_target.transpose(-2, -1)  # (T, N, N)

    # Set diagonal to 0 before adding self-loops
    diag_mask = torch.eye(N, device=device, dtype=torch.bool).unsqueeze(0)
    adj = adj.masked_fill(diag_mask, 0.0)

    # Add identity self-loops (np.eye(N))
    eye = torch.eye(N, device=device).unsqueeze(0)  # (1, N, N)
    adj = adj + eye

    # Row-normalize: each row j sums to 1.0 (D^{-1} A)
    row_sum = adj.sum(dim=-1, keepdim=True)
    adj = adj / torch.clamp(row_sum, min=1e-8)

    if not is_batched:
        return adj.squeeze(0)
    return adj
