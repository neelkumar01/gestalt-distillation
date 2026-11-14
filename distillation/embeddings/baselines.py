import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from transformers import AutoModel
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from tqdm.auto import tqdm


def prepare_frozen_baselines(
    config, splits, dataloaders, student_embeddings_dir, teacher_target_dir
):

    out = Path(student_embeddings_dir)
    split_names = ("train", "val", "test")
    n_features = config["embedding_dim"]

    pretrained_paths = {s: out / f"pretrained_dinov3_{s}.npy" for s in split_names}
    frozen_paths = {s: out / f"frozen_distilled_{s}.npy" for s in split_names}

    if not all(p.exists() for p in pretrained_paths.values()):
        backbone = (
            AutoModel.from_pretrained(config["student_model"])
            .to(config["device"])
            .eval()
        )

        hidden = backbone.config.hidden_size

        for split in split_names:
            path = pretrained_paths[split]
            dataset = dataloaders[split].dataset
            assert np.array_equal(dataset.indices, splits[split])

            if path.exists():
                continue

            tmp = path.with_suffix(".partial.npy")
            if tmp.exists():
                raise FileExistsError(tmp)

            loader = DataLoader(
                dataset,
                batch_size=64,
                shuffle=False,
                num_workers=0,
                pin_memory=True,
            )
            array = np.lib.format.open_memmap(
                tmp,
                mode="w+",
                dtype=np.float32,
                shape=(len(dataset), hidden),
            )

            offset = 0
            with torch.inference_mode():
                for images, _ in tqdm(loader, desc=f"Pretrained · {split}"):
                    images = images.to(config["device"], non_blocking=True)
                    with torch.autocast(
                        "cuda",
                        dtype=torch.bfloat16,
                        enabled=str(config["device"]).startswith("cuda"),
                    ):
                        features = backbone(pixel_values=images).pooler_output

                    batch = features.float().cpu().numpy()
                    assert np.isfinite(batch).all()
                    array[offset : offset + len(batch)] = batch
                    offset += len(batch)

            assert offset == len(dataset)
            array.flush()
            del array
            tmp.rename(path)

        del backbone
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    for split, path in pretrained_paths.items():
        x = np.load(path, mmap_mode="r")
        assert x.shape == (len(splits[split]), 768), (split, x.shape)
        assert np.isfinite(x).all()

    with np.load(teacher_target_dir / "target_normalization.npz") as norm:
        mean, std = norm["mean"], norm["std"]

    teacher = np.load(teacher_target_dir / "gestalt_1024.npy", mmap_mode="r")
    targets = (np.asarray(teacher[splits["train"]], dtype=np.float32) - mean) / std

    probe = make_pipeline(
        StandardScaler(),
        Ridge(alpha=100.0, solver="cholesky"),
    )
    probe.fit(
        np.load(pretrained_paths["train"], mmap_mode="r"),
        targets,
    )

    for split in split_names:
        path = frozen_paths[split]

        if not path.exists():
            tmp = path.with_suffix(".partial.npy")
            if tmp.exists():
                raise FileExistsError(tmp)

            features = np.load(pretrained_paths[split], mmap_mode="r")
            output = np.lib.format.open_memmap(
                tmp,
                mode="w+",
                dtype=np.float32,
                shape=(len(features), n_features),
            )

            for start in range(0, len(features), 1024):
                end = min(start + 1024, len(features))
                output[start:end] = probe.predict(features[start:end])

            output.flush()
            del output
            tmp.rename(path)

        saved = np.load(path, mmap_mode="r")
        assert saved.shape == (len(splits[split]), n_features)
        assert np.isfinite(saved).all()

    print("\n\nBASELINE EMBEDDINGS READY\n")
    print(f"{'Split':<8} {'Pretrained':>18} {'Frozen':>18}")
    for split in split_names:
        a = np.load(pretrained_paths[split], mmap_mode="r")
        b = np.load(frozen_paths[split], mmap_mode="r")
        print(f"{split:<8} {str(a.shape):>18} {str(b.shape):>18}")

    print("\nPretrained: original DINOv3, no fine-tuning")
    print("Frozen: fixed backbone + train-only Ridge projection")

    return pretrained_paths, frozen_paths
