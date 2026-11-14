from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler


def evaluate_tuned_ridge(
    config,
    splits,
    student_embeddings_dir,
    teacher_target_dir,
    pretrained_paths,
    frozen_paths,
):

    root = Path(config["output_dir"])
    bench = root / "benchmarks"
    selection = pd.read_csv(bench / "seed42_exp1_ridge_validation.csv")

    result_path = bench / "seed42_exp1_tuned_test_r2.csv"
    prediction_path = bench / "seed42_exp1_tuned_predictions.npz"
    for path in (result_path, prediction_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite: {path}")

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
    protocols = ("Raw", "Clipped")
    keys = [(p, prop) for p in protocols for prop in properties]
    grid = np.logspace(-1, 7, 17)
    ids = {s: np.asarray(splits[s]) for s in ("train", "val", "test")}
    n = config["num_galaxies"]

    teacher_files = {
        p.stem: p for p in sorted((root / "teacher_embeddings").glob("*.npy"))
    }
    assert len(teacher_files) == 22, "Expected 22 cached teachers."

    sources = {
        **{name: ("full", path) for name, path in teacher_files.items()},
        "Gestalt-22": ("full", teacher_target_dir / "gestalt_1024.npy"),
        "Pretrained DINOv3": ("split", pretrained_paths),
        "Frozen-distilled": ("split", frozen_paths),
        "Student-12": (
            "split",
            {s: student_embeddings_dir / f"student_12_{s}.npy" for s in ids},
        ),
        "Student-22": (
            "split",
            {s: student_embeddings_dir / f"student_22_{s}.npy" for s in ids},
        ),
    }

    models = list(sources)
    assert set(selection["Model"]) == set(models)
    assert len(selection) == len(models) * len(keys)
    assert not selection.duplicated(["Model", "Protocol", "Property"]).any()

    boundary = selection[selection["Alpha_At_Boundary"].astype(bool)]
    lower = boundary[np.isclose(boundary["Selected_Alpha"], grid[0])]
    upper = boundary[np.isclose(boundary["Selected_Alpha"], grid[-1])]

    print(f"Alpha boundary selections: {len(lower)} lower, {len(upper)} upper")
    if len(upper):
        print(
            upper[["Model", "Protocol", "Property", "Selected_Alpha"]].to_string(
                index=False
            )
        )
        raise ValueError(
            "Some alpha selections reached the upper grid limit. "
            "Review validation tuning before test evaluation."
        )

    with np.load(root / "data" / "galaxy_properties.npz") as data:
        labels = np.column_stack([data[p] for p in properties]).astype(np.float64)

    assert labels.shape == (n, 8) and np.isfinite(labels).all()

    lo, hi = np.percentile(labels[ids["train"]], [1, 99], axis=0)
    targets = {"Raw": labels, "Clipped": np.clip(labels, lo, hi)}
    y = {s: np.column_stack([targets[p][ids[s]] for p in protocols]) for s in ids}

    y_mean = y["train"].mean(axis=0)
    y_centered = y["train"] - y_mean
    device = torch.device(config["device"])
    yt = torch.as_tensor(y_centered, dtype=torch.float32, device=device)
    intercept = torch.as_tensor(y_mean, dtype=torch.float32, device=device)

    predictions = np.empty(
        (len(models), len(protocols), len(ids["test"]), len(properties)),
        dtype=np.float32,
    )
    records = []

    for i, (name, (kind, path)) in enumerate(sources.items()):
        selected = (
            selection[selection["Model"] == name]
            .set_index(["Protocol", "Property"])
            .reindex(keys)
        )
        if selected["Selected_Alpha"].isna().any():
            raise ValueError(f"Incomplete alpha selections: {name}")

        chosen = selected["Selected_Alpha"].to_numpy(dtype=np.float64)
        assert np.all(
            np.isclose(chosen[:, None], grid[None, :], rtol=1e-8).any(axis=1)
        ), f"{name}: alpha outside original search grid."

        if kind == "full":
            full = np.load(path, mmap_mode="r")
            assert full.ndim == 2 and full.shape[0] == n
            features = {s: full[ids[s]] for s in ids}
        else:
            features = {s: np.load(path[s], mmap_mode="r") for s in ids}

        d = features["train"].shape[1]
        assert all(
            features[s].shape == (len(ids[s]), d) and np.isfinite(features[s]).all()
            for s in ids
        ), f"{name}: invalid features."

        scaler = StandardScaler()
        xtr = torch.as_tensor(
            np.ascontiguousarray(scaler.fit_transform(features["train"])),
            dtype=torch.float32,
            device=device,
        )
        xv = torch.as_tensor(
            np.ascontiguousarray(scaler.transform(features["val"])),
            dtype=torch.float32,
            device=device,
        )
        xt = torch.as_tensor(
            np.ascontiguousarray(scaler.transform(features["test"])),
            dtype=torch.float32,
            device=device,
        )

        gram = xtr.T @ xtr
        eigenvalues, vectors = torch.linalg.eigh((gram + gram.T) / 2)
        eigenvalues = eigenvalues.clamp_min(0)

        projected_targets = vectors.T @ (xtr.T @ yt)
        alpha_tensor = torch.as_tensor(chosen, dtype=torch.float32, device=device)
        weights = projected_targets / (eigenvalues[:, None] + alpha_tensor[None, :])

        val_pred = ((xv @ vectors) @ weights + intercept).cpu().numpy()
        val_ss = ((y["val"] - y["val"].mean(axis=0)) ** 2).sum(axis=0)
        val_r2 = 1 - ((y["val"] - val_pred) ** 2).sum(axis=0) / val_ss

        assert np.allclose(
            val_r2,
            selected["Validation_R2"].to_numpy(),
            atol=1e-3,
            rtol=0,
        ), f"{name}: validation scores do not reproduce Cell 29."

        test_pred = ((xt @ vectors) @ weights + intercept).cpu().numpy()
        test_ss = ((y["test"] - y["test"].mean(axis=0)) ** 2).sum(axis=0)
        test_r2 = 1 - ((y["test"] - test_pred) ** 2).sum(axis=0) / test_ss

        assert np.isfinite(test_r2).all()

        predictions[i] = test_pred.reshape(
            len(ids["test"]), len(protocols), len(properties)
        ).transpose(1, 0, 2)

        for j, (protocol, prop) in enumerate(keys):
            records.append(
                {
                    "Model": name,
                    "Protocol": protocol,
                    "Property": prop,
                    "Dimensions": d,
                    "Selected_Alpha": chosen[j],
                    "Validation_R2": val_r2[j],
                    "Test_R2": test_r2[j],
                }
            )

        print(f"[{i + 1:02d}/{len(models)}] {name:<20} PASS")

    report = pd.DataFrame(records)

    with result_path.open("x") as f:
        report.to_csv(f, index=False)

    with prediction_path.open("xb") as f:
        np.savez_compressed(
            f,
            predictions=predictions,
            models=np.asarray(models),
            protocols=np.asarray(protocols),
            properties=np.asarray(properties),
            test_indices=ids["test"],
            y_true=np.stack([targets[p][ids["test"]] for p in protocols]),
        )

    overview = report.pivot_table(
        index="Model", columns="Protocol", values="Test_R2", aggfunc="mean"
    ).sort_values("Clipped", ascending=False)

    student = (
        report[(report.Model == "Student-22") & (report.Protocol == "Clipped")]
        .set_index("Property")["Test_R2"]
        .reindex(properties)
    )

    print("\nMean test R² across eight properties:")
    print(overview.round(4).to_string())
    print("\nStudent-22: clipped test R² by property:")
    print(student.round(4).to_string())

    print(f"\nSaved: {result_path}")
    print(f"Saved: {prediction_path}")

    return report
