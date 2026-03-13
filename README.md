## OSA-Sound: Sleep Apnea Detection from Respiratory Sound

This project implements a binary classifier for apnea events from sleep respiratory sound of the following paper:

**“Deep representation learning with cross attention-based multi-feature fusion for sleep apnea detection using sleep respiratory sound”**[`file://1-s2.0-S1746809425016040-main.pdf`](file://1-s2.0-S1746809425016040-main.pdf).

### Features

- **Self-supervised representation learning**: an encoder-decoder is trained to reconstruct raw respiratory waveform, and only the encoder is kept as feature extractor.
- **Multi-modal feature fusion**: encoder features are fused with handcrafted acoustic features and demographic information via hierarchical cross-attention.
- **TabNet classifier**: fused features are fed into a TabNet-based classifier for binary apnea vs. non-apnea prediction.

### Project structure

- `osa_sound/`
  - `models/`
    - `wave_encoder.py`: self-supervised encoder-decoder for raw waveform
    - `fusion.py`: cross-attention based multi-feature fusion
    - `tabnet.py`: TabNet classifier implementation
  - `data/`
    - `dataset.py`: dataset and feature loading wrappers
- `train_ssl.py`: self-supervised reconstruction training utilities (no CLI)
- `train_classifier.py`: downstream binary classification training utilities (no CLI)
- `main.py`: simple entry point placeholder you can customize


### Citation

If you use this repository in your research, please cite the original paper:

> Wang P., Lin Y.-C., Liu W.-T., et al., Deep representation learning with cross attention-based multi-feature fusion for sleep apnea detection using sleep respiratory sound, Biomedical Signal Processing and Control, 113 (2026) 109093.[`file://1-s2.0-S1746809425016040-main.pdf`](file://1-s2.0-S1746809425016040-main.pdf)


