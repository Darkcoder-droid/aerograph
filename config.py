import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Base paths
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
CHECKPOINTS_DIR = ROOT_DIR / "checkpoints"

DATA_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

# Domain & Geospatial Settings
CITY_NAME = "Bengaluru"
CITY_CENTER = (12.9716, 77.5946)  # (latitude, longitude)
RADIUS_M = 25000                   # 25 km radius
DEFAULT_SIGMA_KM = 8.0             # Gaussian spatial decay parameter

# Data Ingestion Settings
HISTORY_DAYS = 90
SEQ_IN = 24                        # Past 24 hours of inputs
SEQ_OUT = 6                        # Next 6 hours of forecasts
OPENAQ_BASE_URL = "https://api.openaq.org/v3"
OPENMETEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
OPENMETEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
CARTO_API_KEY = os.getenv("CARTO_API_KEY", "cb1_5dzp_1_d977400df891f7509d837f65")

# Model Hyperparameters
HIDDEN_DIM = 64
NUM_GCN_LAYERS = 2
NUM_GRU_LAYERS = 2
DROPOUT = 0.1
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 40
HUBER_DELTA = 10.0
