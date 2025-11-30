from typing import Dict, Iterable, List, Tuple
import numpy as np

# Ensure you import these from your actual module structure
from .transient_engine import TransientEngine
from .network_model import NetworkModel


def physics_similarity(observed: np.ndarray, simulated: np.ndarray) -> float:
    """
    Computes similarity score (0 to 1) between observed and simulated traces.
    Uses a normalized MSE approach.
    """
    # Ensure shapes match; truncate if necessary (simple safety check)
    min_len = min(observed.shape[-1], simulated.shape[-1])
    obs_trunc = observed[..., :min_len]
    sim_trunc = simulated[..., :min_len]

    mse = float(np.mean((obs_trunc - sim_trunc) ** 2))
    
    # Avoid divide by zero if signal is flat
    var = float(np.var(obs_trunc))
    denom = var if var > 1e-8 else 1.0
    
    # Score: 1.0 is perfect match, 0.0 is poor match
    score = max(0.0, 1.0 - mse / denom)
    return score


def rank_leak_candidates(
    network: NetworkModel,
    engine: TransientEngine,
    event_type: str,
    element_id: str,
    candidate_node_ids: Iterable[str],
    observed_node_series: Dict[str, Dict[str, List[float]]],
    ts: float = 1.0,
    tc: float = 1.0,
) -> List[Tuple[str, float]]:
    """
    Validates ML candidates by running physics simulations for each.
    
    Args:
        candidate_node_ids: List of Node IDs (strings) suggested by the ML model.
        observed_node_series: The actual sensor data from the field.
    """
    ranked_results = []
    
    # Pre-process observed data into a flat array for easy comparison
    # We only care about nodes that exist in the observation
    sensor_ids = list(observed_node_series.keys())
    
    # Helper to flatten a series dict into a numpy array [1, n_sensors, time]
    def to_array(series_dict):
        # Assumes all sensors have same length, aligned time grid
        data = [series_dict[s]["head"] for s in sensor_ids]
        return np.array(data, dtype=float)

    obs_array = to_array(observed_node_series)

    for node_id in candidate_node_ids:
        try:
            # 1. Run Simulation with leak at this candidate node
            if event_type == "valve_closure":
                # Using a default emitter coeff for verification (or pass it in)
                tm_sim = engine.simulate_valve_closure_with_leak(
                    valve_id=element_id,
                    leak_node_id=node_id,
                    emitter_coeff=0.01, # Standardized test coefficient
                    ts=ts, tc=tc
                )
            else:
                # Fallback or expand for pumps
                continue

            # 2. Extract results for the specific sensors we have data for
            sim_series = engine.extract_node_heads(tm_sim, sensor_ids)
            sim_array = to_array(sim_series)

            # 3. Compute Similarity
            score = physics_similarity(obs_array, sim_array)
            ranked_results.append((node_id, score))

        except Exception as e:
            print(f"[Physics Check] Simulation failed for candidate {node_id}: {e}")
            ranked_results.append((node_id, 0.0))

    # Sort descending by score (highest physics match first)
    ranked_results.sort(key=lambda x: x[1], reverse=True)
    
    return ranked_results