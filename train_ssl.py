from pathlib import Path

import torch
from torch.utils.data import DataLoader

from osa_sound.models.wave_encoder import WaveAutoEncoder, ssl_total_loss


def train_ssl(
    dataloader: DataLoader,
    epochs: int = 50,
    lr: float = 1e-3,
    device: str | None = None,
    save_encoder_path: str | None = "checkpoints/wave_encoder_ssl.pt",
) -> WaveAutoEncoder:
    """Train WaveAutoEncoder in a self-supervised way on your own dataloader.

    Args:
        dataloader: PyTorch DataLoader yielding dicts with key "waveform" -> (B, T) tensor.
        epochs: number of training epochs.
        lr: learning rate.
        device: device string; if None, will choose CUDA if available.
        save_encoder_path: optional path to save encoder weights (set to None to skip saving).

    Returns:
        Trained WaveAutoEncoder model.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = WaveAutoEncoder()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_samples = 0

        for batch in dataloader:
            waveform = batch["waveform"].to(device)
            stage_labels = batch.get("stage_label", None)
            if stage_labels is not None:
                stage_labels = stage_labels.to(device)

            _, x_rec, stage_logits = model(waveform)
            loss = ssl_total_loss(waveform, x_rec, stage_logits, stage_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * waveform.size(0)
            n_samples += waveform.size(0)

        avg_loss = total_loss / max(n_samples, 1)
        print(f"[SSL] Epoch {epoch}/{epochs} - recon_loss={avg_loss:.4f}")

    if save_encoder_path is not None:
        ckpt_path = Path(save_encoder_path)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.encoder.state_dict(), ckpt_path)
        print(f"Saved encoder weights to {ckpt_path}")

    return model

