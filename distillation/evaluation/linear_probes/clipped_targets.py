import json
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error


def benchmark_clipped_targets(
    config, splits, student_embeddings_dir, teacher_target_dir
):
    root = config["output_dir"]
    out = root / "benchmarks"
    csv_path = out / "seed42_validation_probes_clipped.csv"
    metadata_path = out / "seed42_clipping_protocol.json"

    if csv_path.exists() or metadata_path.exists():
        raise FileExistsError("Clipped benchmark already exists.")

    with np.load(root / "data/galaxy_properties.npz") as data:
        names = list(data.files)
        Y_train = np.column_stack([data[k][splits["train"]] for k in names]).astype(
            np.float64
        )
        Y_val = np.column_stack([data[k][splits["val"]] for k in names]).astype(
            np.float64
        )

    assert len(names) == 8
    assert np.isfinite(Y_train).all() and np.isfinite(Y_val).all()

    lower, upper = np.percentile(Y_train, [1, 99], axis=0)
    Y_train = np.clip(Y_train, lower, upper)
    Y_val = np.clip(Y_val, lower, upper)

    gestalt = np.load(teacher_target_dir / "gestalt_1024.npy", mmap_mode="r")

    results = []
    for source in ("Student-12", "Student-22", "Gestalt-22"):
        if source == "Gestalt-22":
            X_train = gestalt[splits["train"]]
            X_val = gestalt[splits["val"]]
        else:
            stage = source.split("-")[1]
            X_train = np.load(
                student_embeddings_dir / f"student_{stage}_train.npy",
                mmap_mode="r",
            )
            X_val = np.load(
                student_embeddings_dir / f"student_{stage}_val.npy",
                mmap_mode="r",
            )

        probe = make_pipeline(
            StandardScaler(),
            Ridge(alpha=100.0, solver="cholesky"),
        )
        probe.fit(X_train, Y_train)
        predictions = probe.predict(X_val)

        for j, name in enumerate(names):
            results.append(
                {
                    "Source": source,
                    "Property": name,
                    "R2": r2_score(Y_val[:, j], predictions[:, j]),
                    "MAE": mean_absolute_error(Y_val[:, j], predictions[:, j]),
                    "N_train": len(X_train),
                    "N_val": len(X_val),
                }
            )

    report = pd.DataFrame(results)
    scores = report.pivot(index="Property", columns="Source", values="R2")[
        ["Student-12", "Student-22", "Gestalt-22"]
    ]

    scores["Δ 22−12"] = scores["Student-22"] - scores["Student-12"]

    with open(csv_path, "x") as f:
        report.to_csv(f, index=False)

    with open(metadata_path, "x") as f:
        json.dump(
            {
                "protocol": "1st/99th-percentile target clipping",
                "percentile_fit_split": "train",
                "evaluation_split": "validation",
                "test_used": False,
                "ridge_alpha": 100.0,
                "thresholds": {
                    name: {"lower": float(lo), "upper": float(hi)}
                    for name, lo, hi in zip(names, lower, upper)
                },
            },
            f,
            indent=2,
        )

    print("\nValidation R² Scores (1st–99th Percentile Target Clipping)\n")
    print(scores.round(4).to_string())
    print("\nStudent-22 improvements:", int((scores["Δ 22−12"] > 0).sum()), "/ 8")
    print("Clipping thresholds: TRAIN ONLY")
    print("Test split: NOT ACCESSED")
    print("Saved:", csv_path)
    print("Saved:", metadata_path)

    return report
