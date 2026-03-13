import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvEncoder(nn.Module):
    """1D convolutional encoder that maps raw waveform to latent representations."""
    def __init__(self, in_channels: int = 1, base_channels: int = 64, latent_dim: int = 256):
        super().__init__()
        channels = [in_channels, base_channels, base_channels * 2, base_channels * 4]
        layers = []
        for i in range(len(channels) - 1):
            layers.append(
                nn.Conv1d(
                    channels[i],
                    channels[i + 1],
                    kernel_size=7,
                    stride=2,
                    padding=3,
                )
            )
            layers.append(nn.BatchNorm1d(channels[i + 1]))
            layers.append(nn.GELU())
        self.conv = nn.Sequential(*layers)
        self.proj = nn.Conv1d(channels[-1], latent_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T) or (B, 1, T)
        return: (B, latent_dim, T_down)
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)
        h = self.conv(x)
        z = self.proj(h)
        return z


class ConvDecoder(nn.Module):
    """1D convolutional decoder that reconstructs waveform from latent representations."""
    def __init__(self, latent_dim: int = 256, base_channels: int = 64, out_channels: int = 1):
        super().__init__()
        channels = [latent_dim, base_channels * 4, base_channels * 2, base_channels, out_channels]
        layers = []
        for i in range(len(channels) - 1):
            stride = 2 if i < len(channels) - 2 else 1
            layers.append(
                nn.ConvTranspose1d(
                    channels[i],
                    channels[i + 1],
                    kernel_size=7,
                    stride=stride,
                    padding=3,
                    output_padding=1 if stride == 2 else 0,
                )
            )
            if i < len(channels) - 2:
                layers.append(nn.BatchNorm1d(channels[i + 1]))
                layers.append(nn.GELU())
        self.deconv = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: (B, latent_dim, T_down)
        return: (B, 1, T_recon)
        """
        x_rec = self.deconv(z)
        return x_rec


class WaveAutoEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 64,
        latent_dim: int = 256,
        stage_hidden_dim: int = 128,
    ):
        super().__init__()
        self.encoder = ConvEncoder(in_channels, base_channels, latent_dim)
        self.decoder = ConvDecoder(latent_dim, base_channels, in_channels)
        self.gru = nn.GRU(input_size=latent_dim, hidden_size=stage_hidden_dim, batch_first=True)
        self.stage_head = nn.Linear(stage_hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        x_rec = self.decoder(z)
        z_seq = z.transpose(1, 2)
        h, _ = self.gru(z_seq)
        h_last = h[:, -1, :]
        stage_logits = self.stage_head(h_last).squeeze(-1)
        return z, x_rec, stage_logits


def stft_reconstruction_loss(
    x: torch.Tensor,
    x_rec: torch.Tensor,
    n_fft: int = 512,
    hop_length: int = 128,
    win_length: int = 512,
) -> torch.Tensor:
    if x.dim() == 3:
        x = x.squeeze(1)
    if x_rec.dim() == 3:
        x_rec = x_rec.squeeze(1)
    X = torch.stft(x, n_fft=n_fft, hop_length=hop_length, win_length=win_length, return_complex=True)
    X_rec = torch.stft(x_rec, n_fft=n_fft, hop_length=hop_length, win_length=win_length, return_complex=True)
    mag = X.abs()
    mag_rec = X_rec.abs()
    l_stft = (mag - mag_rec).pow(2).mean()
    l_log = (torch.log(mag + 1e-7) - torch.log(mag_rec + 1e-7)).pow(2).mean()
    return 0.5 * l_stft + 0.5 * l_log


def ssl_total_loss(
    x: torch.Tensor,
    x_rec: torch.Tensor,
    stage_logits: torch.Tensor | None,
    stage_labels: torch.Tensor | None,
    lambda_recon: float = 0.7,
    lambda_sup: float = 0.3,
) -> torch.Tensor:
    recon = stft_reconstruction_loss(x, x_rec)
    if stage_logits is None or stage_labels is None or (stage_labels < 0).all():
        return recon
    stage_labels = stage_labels.float()
    sup = F.binary_cross_entropy_with_logits(stage_logits, stage_labels)
    return lambda_recon * recon + lambda_sup * sup


