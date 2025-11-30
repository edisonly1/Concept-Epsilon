from pathlib import Path
import sys
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
import os

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.config import SimulationConfig, RUNS_DIR
from app.core.network_model import NetworkModel
from app.core.transient_engine import TransientEngine

#Configuration
SENSORS = ["N2", "N3", "N4"]
LEAK_NODES = ["N5", "N6", "N7", "N8"]
LEAK_COEFFS = [0.005, 0.01, 0.02]
DURATION = 20.0
NOISE_STD = 0.01

def simulate_single_scenario(scenario_id: int, seed: int) -> list[dict]:
    """
    Worker function. Must instantiate its own engine to be process safe.
    """
    
    rng = np.random.default_rng(seed)
    
    sim_cfg = SimulationConfig(duration_seconds=DURATION)
    
    network = NetworkModel.from_name("Tnet1.inp", sim_config=sim_cfg)
    engine = TransientEngine(network_model=network, sim_config=sim_cfg)
    
    leak_present = rng.random() < 0.5
    rows = []

    try:
        if leak_present:
            leak_node = str(rng.choice(LEAK_NODES))
            emitter_coeff = float(rng.choice(LEAK_COEFFS))
            tm = engine.simulate_valve_closure_with_leak(
                valve_id="VALVE",
                leak_node_id=leak_node,
                emitter_coeff=emitter_coeff,
            )
        else:
            leak_node = None
            emitter_coeff = 0.0
            tm = engine.simulate_valve_closure(valve_id="VALVE")

        series = engine.extract_node_heads(tm, SENSORS)

        for nid in SENSORS:
            if nid not in series:
                continue
            t = series[nid]["t"]
            head = np.array(series[nid]["head"], dtype=float)

            # Add noise
            if NOISE_STD > 0:
                head = head + rng.normal(0.0, NOISE_STD, size=head.shape)

            
            for time, value in zip(t, head):
                rows.append({
                    "scenario_id": scenario_id,
                    "leak_present": int(leak_present),
                    "leak_node": leak_node,
                    "node_id": nid,
                    "t": float(time),
                    "head": float(value),
                })
                
    except Exception as e:
        print(f"Scenario {scenario_id} failed: {e}")
        return []

    return rows

def main() -> None:
    n_scenarios = 1500
    max_workers = min(8, os.cpu_count() or 1)
    
    print(f"Generating {n_scenarios} scenarios using {max_workers} workers...")
    
    all_rows = []
    
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for i in range(n_scenarios):
            # Pass a unique seed to each worker to ensure randomness
            seed = 42 + i
            futures.append(executor.submit(simulate_single_scenario, i, seed))
            
        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            all_rows.extend(result)
            if (i + 1) % 50 == 0:
                print(f"Completed {i + 1}/{n_scenarios}")

    if not all_rows:
        print("No data generated.")
        return

    df = pd.DataFrame(all_rows)
    out_path = RUNS_DIR / "leak_dataset_tnet1.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Sort for cleaner reading
    df.sort_values(by=["scenario_id", "node_id", "t"], inplace=True)
    
    df.to_parquet(out_path, index=False)
    print(f"Saved {len(df)} rows to {out_path}")

if __name__ == "__main__":
    try:
        main()
    except RuntimeError as e:
        print(f"Error: {e}")