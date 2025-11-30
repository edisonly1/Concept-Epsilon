import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
import sys

# Ensure we can find paths relative to this script
CURRENT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = CURRENT_DIR.parent
DATA_FILE = BACKEND_DIR / "data" / "runs" / "leak_dataset_tnet1.parquet"

def plot_real_data_moat():
    print(f"Loading dataset from {DATA_FILE}...")
    
    if not DATA_FILE.exists():
        print(f"Error: File not found at {DATA_FILE}")
        print("Please run 'generate_leak_dataset.py' first to create the data.")
        return

    # 1. Load the Data
    df = pd.read_parquet(DATA_FILE)
    print(f"Dataset loaded. Columns found: {list(df.columns)}")
    
    # 2. Filter Data
    # We only want to plot ONE sensor's perspective (e.g., N3) 
    target_sensor = "N3" 
    sensor_df = df[df["node_id"] == target_sensor]
    
    if sensor_df.empty:
        # Fallback if N3 doesn't exist, pick the first sensor available
        unique_sensors = df["node_id"].unique()
        if len(unique_sensors) > 0:
            target_sensor = unique_sensors[0]
            sensor_df = df[df["node_id"] == target_sensor]
            print(f"Sensor 'N3' not found. Switching to '{target_sensor}'.")
        else:
            print("No sensor data found.")
            return

    # 3. Select Scenarios
    # Check if 'label' column exists to filter for leaks
    if "label" in sensor_df.columns:
        # Filter for leaks (assuming '0' is baseline/no-leak)
        leak_scenarios = sensor_df[sensor_df["label"] != "0"]["scenario_id"].unique()
        print(f"Found {len(leak_scenarios)} leak scenarios based on 'label'.")
    else:
        # Fallback: Just take unique scenarios if label is missing
        print("Warning: 'label' column not found. Using all available scenarios.")
        leak_scenarios = sensor_df["scenario_id"].unique()
    
    # Select top 50 scenarios to stack
    num_traces = 50
    if len(leak_scenarios) > num_traces:
        # If we have too many, try to skip a few to get variety (e.g., every 5th one)
        # This helps if the first 20 are all identical baselines
        step = max(1, len(leak_scenarios) // num_traces)
        selected_scenarios = leak_scenarios[::step][:num_traces]
    else:
        selected_scenarios = leak_scenarios
    
    print(f"Plotting {len(selected_scenarios)} traces from sensor {target_sensor}...")

    # 4. Setup the Plot
    fig, ax = plt.subplots(figsize=(12, 10), dpi=300)
    ax.set_facecolor('#0f172a') # Dark/Tech background
    fig.patch.set_facecolor('#0f172a')
    
    cmap = cm.get_cmap('magma') 
    
    # Calculate offset based on the data range
    if not sensor_df.empty:
        y_min, y_max = sensor_df["head"].min(), sensor_df["head"].max()
        trace_height = y_max - y_min
        if trace_height == 0: trace_height = 1.0 # Prevent zero division
        offset_step = trace_height * 0.3 
    else:
        offset_step = 1.0

    for i, scen_id in enumerate(selected_scenarios):
        # Extract single trace
        trace = sensor_df[sensor_df["scenario_id"] == scen_id].sort_values("t")
        
        t = trace["t"].values
        head = trace["head"].values
        
        if len(head) == 0: continue

        # Color logic
        color = cmap(0.2 + 0.6 * (i / max(1, len(selected_scenarios))))
        
        # Plot with offset
        # Normalize baseline to 0 for the plot
        normalized_head = head - head.min()
        ax.plot(t, normalized_head + (i * offset_step), color=color, linewidth=1.5, alpha=0.8)

    # 5. Styling
    ax.set_title(f"Concept Epsilon: Real Physics Data Moat (Sensor {target_sensor})", 
                 color='white', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel("Simulation Time (s)", color='#94a3b8', fontsize=12)
    ax.set_ylabel(f"Stacked Pressure Traces ({len(selected_scenarios)} Scenarios)", color='#94a3b8', fontsize=12)
    
    # Custom Grid
    ax.grid(True, which='major', color='#334155', linestyle='--', alpha=0.3)
    
    # Remove boxy spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color('#94a3b8')
    
    ax.tick_params(axis='x', colors='#94a3b8')
    ax.set_yticks([]) # Hide Y values
    
    plt.tight_layout()
    
    output_path = BACKEND_DIR / "real_data_moat.png"
    plt.savefig(output_path, facecolor=fig.get_facecolor())
    print(f"Graph saved to {output_path}")
    plt.show()

if __name__ == "__main__":
    plot_real_data_moat()