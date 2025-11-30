from pathlib import Path
import sys

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))


from app.config import MODELS_DIR
from ml.data import LeakScenarioDataset
from ml.models import LeakCNN


def train() -> None:
    print("Loading dataset...")
    dataset = LeakScenarioDataset()  # uses leak_dataset_tnet1.parquet by default

    n_total = len(dataset)
    if n_total < 2:
        raise RuntimeError(
            f"Not enough scenarios ({n_total}) to train. "
            "Regenerate the dataset with more baseline/leak realizations."
        )

    n_val = max(1, int(0.2 * n_total))
    n_train = n_total - n_val

    train_ds, val_ds = random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = LeakCNN(
        in_channels=len(dataset.sensors),
        num_classes=dataset.num_classes,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Training Configuration
    num_epochs = 50
    training_noise_std = 0.005  # Adds noise for robustness

    print(f"Starting training for {num_epochs} epochs...")

    for epoch in range(num_epochs):
        #Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for X, y in train_loader:
            X = X.to(device)  # [B, C, T]
            y = y.to(device)

            # Data Augmentation: Add random noise during training
            if training_noise_std > 0:
                noise = torch.randn_like(X) * training_noise_std
                X = X + noise

            optimizer.zero_grad()
            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * X.size(0)
            preds = logits.argmax(dim=1)
            train_correct += (preds == y).sum().item()
            train_total += X.size(0)

        train_loss /= train_total
        train_acc = train_correct / train_total

        #Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for X, y in val_loader:
                X = X.to(device)
                y = y.to(device)

                logits = model(X)
                loss = criterion(logits, y)

                val_loss += loss.item() * X.size(0)
                preds = logits.argmax(dim=1)
                val_correct += (preds == y).sum().item()
                val_total += X.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total

        print(
            f"Epoch {epoch + 1:02d}/{num_epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}"
        )

    # Save checkpoint
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = MODELS_DIR / "leak_cnn_tnet1.pt"

    checkpoint = {
        "state_dict": model.state_dict(),
        "in_channels": len(dataset.sensors),
        "num_classes": dataset.num_classes,
        "sensors": dataset.sensors,
        "class_mapping": dataset.class_mapping,
        "time_grid": dataset.time_grid,
        "mean": dataset.mean,
        "std": dataset.std,
    }

    torch.save(checkpoint, out_path)
    print(f"Saved model to {out_path}")


if __name__ == "__main__":
    train()