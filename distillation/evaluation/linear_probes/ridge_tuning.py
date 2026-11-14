from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler


def tune_ridge_alpha(
    config,
    splits,
    student_embeddings_dir,
    teacher_target_dir,
    pretrained_paths,
    frozen_paths,
):

    root = Path(config["output_dir"])
    output = root / "benchmarks" / "seed42_exp1_ridge_validation.csv"
    if output.exists():
        raise FileExistsError(f"Will not overwrite: {output}")

    properties = [
        "Redshift",
        "Stellar mass",
        "sSFR",
        "SFR",
        "Stellar age",
        "HSC g-r color",
        "Bulge radius",
        "Disk radius",
    ]
    alphas = np.logspace(-1, 7, 17)
    train, val = np.asarray(splits["train"]), np.asarray(splits["val"])
    n = config["num_galaxies"]

    with np.load(root / "data" / "galaxy_properties.npz") as data:
        labels = np.column_stack([data[p] for p in properties])

    assert labels.shape == (n, 8)
    assert np.isfinite(labels[np.r_[train, val]]).all()

    lower, upper = np.percentile(labels[train], [1, 99], axis=0)
    protocols = {
        "Raw": labels,
        "Clipped": np.clip(labels, lower, upper),
    }
    y_train = np.column_stack([y[train] for y in protocols.values()])
    y_val = np.column_stack([y[val] for y in protocols.values()])
    y_mean = y_train.mean(axis=0)
    y_centered = y_train - y_mean
    val_ss = ((y_val - y_val.mean(axis=0)) ** 2).sum(axis=0)
    assert np.all(val_ss > 0)

    teacher_paths = {
        p.stem: p for p in sorted((root / "teacher_embeddings").glob("*.npy"))
    }
    assert (
        len(teacher_paths) == 22
    ), f"Expected 22 teacher caches, found {len(teacher_paths)}"

    sources = {
        **{name: ("full", path) for name, path in teacher_paths.items()},
        "Gestalt-22": ("full", teacher_target_dir / "gestalt_1024.npy"),
        "Pretrained DINOv3": ("split", pretrained_paths),
        "Frozen-distilled": ("split", frozen_paths),
        "Student-12": (
            "split",
            {
                s: student_embeddings_dir / f"student_12_{s}.npy"
                for s in ("train", "val")
            },
        ),
        "Student-22": (
            "split",
            {
                s: student_embeddings_dir / f"student_22_{s}.npy"
                for s in ("train", "val")
            },
        ),
    }
    assert len(sources) == 27

    device = torch.device(config["device"])
    targets = torch.as_tensor(y_centered, dtype=torch.float32, device=device)
    intercept = torch.as_tensor(y_mean, dtype=torch.float32, device=device)

    records = []

    for number, (name, (kind, path)) in enumerate(sources.items(), 1):
        if kind == "full":
            x = np.load(path, mmap_mode="r")
            assert x.ndim == 2 and x.shape[0] == n
            x_train, x_val = x[train], x[val]
        else:
            x_train = np.load(path["train"], mmap_mode="r")
            x_val = np.load(path["val"], mmap_mode="r")

        assert x_train.ndim == 2
        assert x_train.shape[0] == len(train)
        assert x_val.shape == (len(val), x_train.shape[1])
        assert np.isfinite(x_train).all() and np.isfinite(x_val).all()

        scaler = StandardScaler()
        xt = torch.as_tensor(
            np.ascontiguousarray(scaler.fit_transform(x_train)),
            dtype=torch.float32,
            device=device,
        )
        xv = torch.as_tensor(
            np.ascontiguousarray(scaler.transform(x_val)),
            dtype=torch.float32,
            device=device,
        )

        gram = xt.T @ xt
        gram = (gram + gram.T) / 2
        eigenvalues, vectors = torch.linalg.eigh(gram)
        eigenvalues = eigenvalues.clamp_min(0)

        projected_y = vectors.T @ (xt.T @ targets)
        projected_val = xv @ vectors

        scores = []
        for alpha in alphas:
            weights = projected_y / (eigenvalues[:, None] + alpha)
            prediction = (
                (projected_val @ weights + intercept).cpu().numpy().astype(np.float64)
            )

            scores.append(1 - ((y_val - prediction) ** 2).sum(axis=0) / val_ss)

        scores = np.asarray(scores)
        assert np.isfinite(scores).all(), f"{name}: invalid validation R²"
        best = np.argmax(scores, axis=0)

        for j, (protocol, property_name) in enumerate(
            (p, prop) for p in protocols for prop in properties
        ):
            records.append(
                {
                    "Model": name,
                    "Protocol": protocol,
                    "Property": property_name,
                    "Dimensions": x_train.shape[1],
                    "Selected_Alpha": float(alphas[best[j]]),
                    "Validation_R2": float(scores[best[j], j]),
                    "Alpha_At_Boundary": bool(best[j] in (0, len(alphas) - 1)),
                }
            )

        print(
            f"[{number:02d}/27] {name:<20} "
            f"clipped mean R²: "
            f"{scores[best, np.arange(16)][8:].mean():.4f}"
        )

        del xt, xv, gram, eigenvalues, vectors, projected_y, projected_val
        if device.type == "cuda":
            torch.cuda.empty_cache()

    report = pd.DataFrame(records)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as file:
        report.to_csv(file, index=False)

    clipped = report[report["Protocol"] == "Clipped"]
    overview = (
        clipped.groupby("Model")["Validation_R2"].mean().sort_values(ascending=False)
    )

    print("\nVALIDATION TUNING COMPLETE")
    print(f"Models: {report.Model.nunique()} | Properties: 8")
    print(f"Protocols: Raw + Clipped | Probes selected: {len(report)}")
    print(f"Alpha grid: {alphas[0]:g} to {alphas[-1]:g}")
    print("Boundary selections:", int(report.Alpha_At_Boundary.sum()))
    print("\nTop 10 — mean clipped validation R²")
    print(overview.head(10).round(4).to_string())
    print("\nStudent-22:", round(overview["Student-22"], 4))
    print("\nSaved:", output)

    return report
