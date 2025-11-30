from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from app.config import RUNS_DIR


class LeakScenarioDataset(Dataset):
    """
    Loads leak_dataset_tnet1.parquet and converts it into per scenario
    tensors of shape [channels, time], where channels = sensors.

    Labels are a single multi-class target:
      0 -> no leak
      1..K -> leak at leak_nodes[i-1]
    """

    def __init__(
        self,
        parquet_path: Path | None = None,
        sensors: List[str] | None = None,
    ) -> None:
        if parquet_path is None:
            parquet_path = RUNS_DIR / "leak_dataset_tnet1.parquet"

        if not parquet_path.exists():
            raise FileNotFoundError(f"Parquet dataset not found: {parquet_path}")

        self.parquet_path = parquet_path
        df = pd.read_parquet(parquet_path)

        # Sensors: default to all distinct node_ids in the dataset
        if sensors is None:
            sensors = sorted(df["node_id"].unique().tolist())
        self.sensors: List[str] = sensors

        # Time grid: assume same for all scenarios, sort once
        self.time_grid: List[float] = sorted(df["t"].unique().tolist())

        # Build class mapping: "none" + sorted leak nodes
        leak_nodes = sorted(df["leak_node"].dropna().unique().tolist())
        class_names = ["none"] + leak_nodes
        self.class_mapping: Dict[str, int] = {
            name: idx for idx, name in enumerate(class_names)
        }
        self.idx_to_class: Dict[int, str] = {
            idx: name for name, idx in self.class_mapping.items()
        }
        self.num_classes: int = len(class_names)

        # Build X, y arrays
        self._build_arrays(df)

    def _build_arrays(self, df: pd.DataFrame) -> None:
        X_list: list[np.ndarray] = []
        y_list: list[int] = []

        for scenario_id, g in df.groupby("scenario_id"):
            # Filter to chosen sensors
            g_sens = g[g["node_id"].isin(self.sensors)].copy()

            # Pivot: rows = node_id, cols = t, values = head
            piv = g_sens.pivot(index="node_id", columns="t", values="head")

            # Enforce consistent order and fill any missing entries with 0
            piv = piv.reindex(index=self.sensors, columns=self.time_grid)
            arr = piv.values.astype("float32")
            arr = np.nan_to_num(arr, nan=0.0)

            X_list.append(arr)

            # Determine label for this scenario
            leak_vals = g["leak_node"].dropna().unique()
            if len(leak_vals) == 0:
                label_name = "none"
            else:
                assert len(leak_vals) == 1, f"Multiple leak nodes in scenario {scenario_id}"
                label_name = leak_vals[0]

            y_list.append(self.class_mapping[label_name])

        # Raw stacked array: [N_scenarios, C, T]
        self.X: np.ndarray = np.stack(X_list, axis=0)
        self.y: np.ndarray = np.array(y_list, dtype="int64")

        # Per sensor normalization across scenarios and time
        # mean/std shapes: [C]
        self.mean: np.ndarray = self.X.mean(axis=(0, 2))
        self.std: np.ndarray = self.X.std(axis=(0, 2)) + 1e-6

        # Normalize inplace: [N, C, T]
        self.X = (self.X - self.mean[None, :, None]) / self.std[None, :, None]

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(self.X[idx])          # [C, T]
        y = torch.tensor(self.y[idx], dtype=torch.long)
        return x, y
