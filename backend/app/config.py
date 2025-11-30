from dataclasses import dataclass
from pathlib import Path

# Repo root
BASE_DIR = Path(__file__).resolve().parents[2]

# Data dirs
DATA_DIR = BASE_DIR / "backend" / "data"
NETWORKS_DIR = DATA_DIR / "networks"
MODELS_DIR = DATA_DIR / "models"
RUNS_DIR = DATA_DIR / "runs"

for d in (DATA_DIR, NETWORKS_DIR, MODELS_DIR, RUNS_DIR):
    d.mkdir(parents=True, exist_ok=True)


@dataclass
class SimulationConfig:
    """
    Default simulation settings for TSNet runs.

    These are tuned to the Tnet1 example (valve closure) but work for other
    networks as well.
    """

    wave_speed: float = 1200.0          # m/s
    duration_seconds: float = 20.0      # total simulation horizon [s]
    time_step: float = 0.1              # [s] time step
    engine: str = "DD"                  # WNTR engine for initial conditions
