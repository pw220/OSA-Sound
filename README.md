## OSA-Sound: Sleep Apnea Detection from Respiratory Sound

This project implements a binary classifier for apnea events from sleep respiratory sound, following the paper： 
**“Deep representation learning with cross attention-based multi-feature fusion for sleep apnea detection using sleep respiratory sound”**[`file://1-s2.0-S1746809425016040-main.pdf`](file://1-s2.0-S1746809425016040-main.pdf).

### Features

- **Supervised autoencoder for representation learning**
  - Waveform autoencoder trained with **STFT + log-STFT reconstruction loss** for robust spectral reconstruction.
  - A GRU-based head predicts **REM vs. non-REM** sleep stage with binary cross-entropy, forming a **supervised autoencoder**.
- **Hierarchical cross-attention fusion (HCAF)**:
  - Acoustic feature `a`, demographic feature `d`, and latent representation `z` are projected to a shared embedding space and refined via **two-stage cross-attention**:
    1. Acoustic–demographic cross-attention: acoustic as query, demographic as key/value.
    2. Latent–acoustic cross-attention: latent representation as query, refined acoustic as key/value.
  - Residual connections and LayerNorm are applied to preserve original information while integrating complementary context.
- **TabNet classifier**: fused features are fed into a TabNet-based classifier for binary apnea vs. non-apnea prediction on each segment.

### Project structure

- `osa_sound/`
  - `models/`
    - `wave_encoder.py`: self-supervised encoder-decoder for raw waveform
    - `fusion.py`: cross-attention based multi-feature fusion
    - `tabnet.py`: TabNet classifier implementation
  - `data/`
    - `dataset.py`: dataset and feature loading wrappers (`SampleItem` with waveform, acoustic, demo, apnea label, and optional REM/non-REM stage label)
- `train_ssl.py`: self-supervised (supervised autoencoder) training utilities for the waveform encoder
- `train_classifier.py`: downstream apnea event classification training utilities (fusion + TabNet)
- `main.py`: example end-to-end pipeline (SSL pretraining followed by downstream classifier training, using synthetic data)

### Installation

The project targets Python 3.9+ and PyTorch. Install dependencies via:

```bash
pip install -r requirements.txt
```

### Citation

If you use this repository in your research, please cite the original paper:

> Wang P., Lin Y.-C., Liu W.-T., et al., Deep representation learning with cross attention-based multi-feature fusion for sleep apnea detection using sleep respiratory sound, Biomedical Signal Processing and Control, 113 (2026) 109093.[`file://1-s2.0-S1746809425016040-main.pdf`](file://1-s2.0-S1746809425016040-main.pdf)


