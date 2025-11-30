from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import torch

from ..config import MODELS_DIR
from ml.models import LeakCNN


class LeakLocalizer:
    """
    Wrapper around the trained LeakCNN model.

    Expects inputs of shape [batch, channels, time] with channels ordered
    according to `self.sensors`. Applies the same per-sensor normalization
    as used in training.
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        device: Optional[str] = None,
    ) -> None:
        self.model_path = model_path or (MODELS_DIR / "leak_cnn_tnet1.pt")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model: Optional[torch.nn.Module] = None
        self.sensors: Optional[list[str]] = None
        self.class_mapping: Optional[Dict[str, int]] = None
        self.idx_to_class: Optional[Dict[int, str]] = None
        self.time_grid: Optional[list[float]] = None
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None

        self._load()

    def _load(self) -> None:
        if not self.model_path.exists():
            print(f"[LeakLocalizer] No model found at {self.model_path}, using stub.")
            self.model = None
            return

        # Explicitly disable weights_only so we can load numpy arrays (mean/std)
        try:
            checkpoint: Dict[str, Any] = torch.load(
                self.model_path,
                map_location=self.device,
                weights_only=False,  # for PyTorch >= 2.4
            )
        except TypeError:
            # For older PyTorch versions that don't support weights_only
            checkpoint = torch.load(self.model_path, map_location=self.device)

        in_channels = checkpoint["in_channels"]
        num_classes = checkpoint["num_classes"]

        model = LeakCNN(in_channels=in_channels, num_classes=num_classes)
        model.load_state_dict(checkpoint["state_dict"])
        model.to(self.device)
        model.eval()

        self.model = model
        self.sensors = checkpoint.get("sensors")
        self.class_mapping = checkpoint.get("class_mapping")
        self.idx_to_class = {
            idx: name for name, idx in (self.class_mapping or {}).items()
        }
        self.time_grid = checkpoint.get("time_grid")

        mean = checkpoint.get("mean", None)
        std = checkpoint.get("std", None)
        if mean is not None:
            self.mean = np.array(mean, dtype=np.float32)
        if std is not None:
            self.std = np.array(std, dtype=np.float32)

        print(
            f"[LeakLocalizer] Loaded model from {self.model_path} "
            f"with sensors={self.sensors} classes={self.class_mapping}"
        )

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """
        x: np.ndarray of shape [batch, channels, time]

        Returns:
            probs: np.ndarray of shape [batch, num_classes]
        """
        if self.model is None:
            raise RuntimeError(
                "LeakLocalizer model not loaded. Train and save a model first."
            )

        self.model.eval()
        with torch.no_grad():
            tensor = torch.from_numpy(x).float().to(self.device)  # [B, C, T]

            # Apply same per-sensor normalization as training
            if self.mean is not None and self.std is not None:
                mean = torch.as_tensor(
                    self.mean, dtype=tensor.dtype, device=self.device
                )[None, :, None]  # [1, C, 1]
                std = torch.as_tensor(
                    self.std, dtype=tensor.dtype, device=self.device
                )[None, :, None]
                tensor = (tensor - mean) / (std + 1e-6)

            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=-1)
            return probs.cpu().numpy()
