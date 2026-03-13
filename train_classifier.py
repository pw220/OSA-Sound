from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from osa_sound.models.fusion import MultiFeatureFusion
from osa_sound.models.tabnet import TabNetClassifier
from osa_sound.models.wave_encoder import ConvEncoder


class OSAClassifierModel(nn.Module):
    """End-to-end model combining:

    - Wave encoder (frozen or fine-tuned)
    - MultiFeatureFusion
    - TabNetClassifier
    """

    def __init__(
        self,
        dim_enc: int,
        dim_acoustic: int,
        dim_demo: int,
        dim_fused: int = 256,
        tabnet_dim: int = 64,
    ):
        super().__init__()
        self.encoder = ConvEncoder(in_channels=1, base_channels=64, latent_dim=dim_enc)
        self.fusion = MultiFeatureFusion(
            dim_enc=dim_enc,
            dim_acoustic=dim_acoustic,
            dim_demo=dim_demo,
            dim_hidden=dim_fused,
            dim_fused=dim_fused,
        )
        self.tabnet = TabNetClassifier(input_dim=dim_fused, output_dim=2, n_d=tabnet_dim, n_a=tabnet_dim)

    def forward(self, waveform: torch.Tensor, acoustic: torch.Tensor, demo: torch.Tensor) -> torch.Tensor:
        z = self.encoder(waveform)
        fused = self.fusion(z, acoustic, demo)
        logits = self.tabnet(fused)
        return logits


def train_classifier(
    model: OSAClassifierModel,
    dataloader: DataLoader,
    epochs: int = 50,
    lr: float = 1e-3,
    device: str | None = None,
    freeze_encoder: bool = False,
    encoder_ckpt: str | None = None,
    save_model_path: str | None = "checkpoints/osa_classifier.pt",
) -> OSAClassifierModel:
    """Train OSAClassifierModel on your own dataloader.

    Args:
        model: instance of OSAClassifierModel.
        dataloader: PyTorch DataLoader yielding dicts with keys
            "waveform", "acoustic", "demo", "label".
        epochs: number of training epochs.
        lr: learning rate.
        device: device string; if None, will choose CUDA if available.
        freeze_encoder: whether to freeze encoder weights during training.
        encoder_ckpt: optional path to pretrained encoder weights to load before training.
        save_model_path: optional path to save the whole model (set to None to skip saving).

    Returns:
        Trained OSAClassifierModel.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if encoder_ckpt is not None:
        ckpt_path = Path(encoder_ckpt)
        if ckpt_path.exists():
            state = torch.load(ckpt_path, map_location="cpu")
            model.encoder.load_state_dict(state)
            print(f"Loaded encoder weights from {ckpt_path}")
        else:
            print(f"Warning: encoder checkpoint {ckpt_path} not found, training encoder from scratch.")

    if freeze_encoder:
        for p in model.encoder.parameters():
            p.requires_grad = False

    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        n = 0

        for batch in dataloader:
            waveform = batch["waveform"].to(device)
            acoustic = batch["acoustic"].to(device)
            demo = batch["demo"].to(device)
            labels = batch["label"].to(device)

            logits = model(waveform, acoustic, demo)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            n += labels.size(0)

        avg_loss = total_loss / max(n, 1)
        acc = correct / max(n, 1)
        print(f"[CLS] Epoch {epoch}/{epochs} - loss={avg_loss:.4f} acc={acc:.4f}")

    if save_model_path is not None:
        ckpt_path = Path(save_model_path)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), ckpt_path)
        print(f"Saved classifier weights to {ckpt_path}")

    return model

