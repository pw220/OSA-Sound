import numpy as np
from torch.utils.data import DataLoader

from osa_sound.data.dataset import ApneaSoundDataset, SampleItem
from train_ssl import train_ssl
from train_classifier import OSAClassifierModel, train_classifier


def build_ssl_loader(num_samples: int = 256, length: int = 8000, batch_size: int = 32) -> DataLoader:
    samples = {}
    for i in range(num_samples):
        waveform = np.random.randn(length).astype("float32")
        acoustic = np.zeros(1, dtype="float32")
        demo = np.zeros(1, dtype="float32")
        stage_label = np.random.randint(0, 2)
        samples[str(i)] = SampleItem(
            waveform=waveform,
            acoustic=acoustic,
            demo=demo,
            label=None,
            stage_label=int(stage_label),
        )
    dataset = ApneaSoundDataset(samples)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def build_cls_loader(
    num_samples: int = 256,
    length: int = 8000,
    dim_acoustic: int = 64,
    dim_demo: int = 8,
    batch_size: int = 32,
) -> DataLoader:
    samples = {}
    for i in range(num_samples):
        waveform = np.random.randn(length).astype("float32")
        acoustic = np.random.randn(dim_acoustic).astype("float32")
        demo = np.random.randn(dim_demo).astype("float32")
        label = np.random.randint(0, 2)
        samples[str(i)] = SampleItem(waveform=waveform, acoustic=acoustic, demo=demo, label=int(label))
    dataset = ApneaSoundDataset(samples)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def main() -> None:
    ssl_loader = build_ssl_loader()
    ssl_model = train_ssl(ssl_loader, epochs=5, lr=1e-3)

    dim_enc = ssl_model.encoder.proj.out_channels
    dim_acoustic = 64
    dim_demo = 8

    cls_loader = build_cls_loader(dim_acoustic=dim_acoustic, dim_demo=dim_demo)
    model = OSAClassifierModel(dim_enc=dim_enc, dim_acoustic=dim_acoustic, dim_demo=dim_demo)
    train_classifier(
        model=model,
        dataloader=cls_loader,
        epochs=5,
        lr=1e-3,
        encoder_ckpt=None,
        freeze_encoder=False,
    )


if __name__ == "__main__":
    main()

