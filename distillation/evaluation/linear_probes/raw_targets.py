import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error


def benchmark_student_vs_gestalt(
    config, splits, student_embeddings_dir, teacher_target_dir
):

    root = config["output_dir"]
    out = root / "benchmarks"
    out.mkdir(parents=True, exist_ok=True)

    csv_path = out / "seed42_validation_probes.csv"
    fig_path = out / "seed42_validation_r2.png"
    method_path = out / "seed42_probe_method.json"

    if any(p.exists() for p in (csv_path, fig_path, method_path)):
        raise FileExistsError("Benchmark artifacts already exist.")

    with np.load(root / "data/galaxy_properties.npz") as data:
        names = list(data.files)
        labels = {k: np.asarray(data[k], dtype=np.float64) for k in names}

    assert len(names) == 8
    for name, values in labels.items():
        assert values.shape == (config["num_galaxies"],), name
        assert np.isfinite(values[splits["train"]]).all(), name
        assert np.isfinite(values[splits["val"]]).all(), name

    Y_train = np.column_stack([labels[name][splits["train"]] for name in names])
    Y_val = np.column_stack([labels[name][splits["val"]] for name in names])

    sources = {
        "Student-12": (
            student_embeddings_dir / "student_12_train.npy",
            student_embeddings_dir / "student_12_val.npy",
        ),
        "Student-22": (
            student_embeddings_dir / "student_22_train.npy",
            student_embeddings_dir / "student_22_val.npy",
        ),
    }

    gestalt = np.load(teacher_target_dir / "gestalt_1024.npy", mmap_mode="r")
    results = []

    for source in ("Student-12", "Student-22", "Gestalt-22"):
        if source == "Gestalt-22":
            X_train = gestalt[splits["train"]]
            X_val = gestalt[splits["val"]]
        else:
            train_path, val_path = sources[source]
            X_train = np.load(train_path, mmap_mode="r")
            X_val = np.load(val_path, mmap_mode="r")

        assert X_train.shape == (36_000, 1024)
        assert X_val.shape == (4_500, 1024)

        probe = make_pipeline(
            StandardScaler(),
            Ridge(alpha=100.0, solver="cholesky"),
        )
        probe.fit(X_train, Y_train)
        predicted = probe.predict(X_val)

        for j, name in enumerate(names):
            results.append(
                {
                    "Source": source,
                    "Property": name,
                    "R2": r2_score(Y_val[:, j], predicted[:, j]),
                    "MAE": mean_absolute_error(Y_val[:, j], predicted[:, j]),
                    "N_train": len(X_train),
                    "N_val": len(X_val),
                }
            )

    report = pd.DataFrame(results)
    scores = report.pivot(index="Property", columns="Source", values="R2")
    scores = scores[["Student-12", "Student-22", "Gestalt-22"]]
    scores["Δ 22−12"] = scores["Student-22"] - scores["Student-12"]

    print("\nValidation R² Scores: Student vs. Teacher Embeddings\n")
    print(scores.round(4).to_string())
    print("Higher R² is better\nMAE is recorded in CSV")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    matrix = scores[["Student-12", "Student-22", "Gestalt-22"]]
    im = ax.imshow(matrix.to_numpy(), aspect="auto", cmap="RdYlBu", vmin=-0.2, vmax=1.0)
    ax.set_xticks(range(3), matrix.columns)
    ax.set_yticks(range(len(matrix)), matrix.index)
    for i in range(len(matrix)):
        for j in range(3):
            ax.text(
                j, i, f"{matrix.iloc[i, j]:.3f}", ha="center", va="center", fontsize=9
            )
    ax.set_title("Validation R² Scores: Student vs. Teacher Embeddings")
    fig.colorbar(im, ax=ax, label="R²", shrink=0.8)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=160)
    plt.show()
    plt.close(fig)

    with open(csv_path, "x") as f:
        report.to_csv(f, index=False)

    with open(method_path, "x") as f:
        json.dump(
            {
                "split": "train → validation",
                "test_used": False,
                "ridge_alpha": 100.0,
                "feature_normalization": "train-only StandardScaler",
                "sources": list(sources) + ["Gestalt-22"],
                "metrics": ["R2", "MAE"],
            },
            f,
            indent=2,
        )

    print("\nSaved:", csv_path)
    print("Saved:", fig_path)
    return report
