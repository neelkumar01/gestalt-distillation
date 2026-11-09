import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm


def whiten_teacher_embeddings(config, splits, teacher_dir):

    output = config["output_dir"] / "whitened_embeddings"
    output.mkdir(parents=True, exist_ok=True)

    paths = sorted(teacher_dir.glob("*.npy"))
    assert len(paths) == 22

    train = splits["train"]
    records = []

    torch.set_float32_matmul_precision("highest")

    for path in tqdm(paths, desc="PCA + Z-score"):
        name = path.stem
        target = output / f"{name}.npy"
        params = output / f"{name}_pca.npz"

        if target.exists() or params.exists():
            raise FileExistsError(f"{name}: Existing PCA output.")

        X = np.load(path, mmap_mode="r")
        d = X.shape[1]

        T = torch.tensor(X[train], device="cuda")
        center = T.mean(0)
        T -= center

        covariance = (T.T @ T) / (len(train) - 1)
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance)

        eigenvalues = eigenvalues.flip(0)
        components = eigenvectors.flip(1).contiguous()

        projected = T @ components
        mean = projected.mean(0)
        std = projected.std(0, unbiased=False)

        valid = eigenvalues > eigenvalues[0] * 1e-8
        scale = torch.where(valid, std.clamp_min(1e-6), 1.0)

        temp = output / f"{name}.partial.npy"
        if temp.exists():
            raise FileExistsError(temp)

        Z = np.lib.format.open_memmap(temp, mode="w+", dtype=np.float32, shape=X.shape)

        for start in range(0, len(X), 1024):
            end = min(start + 1024, len(X))
            batch = torch.tensor(X[start:end], device="cuda")
            features = ((batch - center) @ components - mean) / scale
            features[:, ~valid] = 0
            Z[start:end] = features.cpu().numpy()

        Z.flush()
        del Z

        np.savez(
            params,
            pca_components=components.T.cpu().numpy(),
            pca_mean=center.cpu().numpy(),
            zscore_mu=mean.cpu().numpy(),
            zscore_sd=scale.cpu().numpy(),
            valid_components=valid.cpu().numpy(),
        )
        temp.rename(target)

        records.append(
            {
                "Teacher": name,
                "Dimensions": d,
                "Retained PCs": int(valid.sum().item()),
                "Suppressed PCs": int((~valid).sum().item()),
            }
        )

        del T, covariance, projected, components, eigenvectors
        torch.cuda.empty_cache()

    report = pd.DataFrame(records)
    report.to_csv(output / "pca_summary.csv", index=False)

    print("\nTRAIN ONLY PCA SUMMARY\n")
    print(report.to_string(index=False))
    print(f"\nCompleted: {len(report)}/22 teachers")
    print("Train only fit: YES")
    print("Saved:", output)

    return output, [p.stem for p in paths]
