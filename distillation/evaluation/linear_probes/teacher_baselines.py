import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from tqdm.auto import tqdm
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

def benchmark_individual_teachers(config, splits, whitened_dir, model_names):

    root = config["output_dir"]
    out = root / "benchmarks"
    result_path = out / "individual_teachers_clipped_val.csv"
    plot_path = out / "teacher_baseline_comparison.png"
    method_path = out / "individual_teachers_method.json"

    if any(p.exists() for p in (result_path, plot_path, method_path)):
        raise FileExistsError("Individual-teacher benchmark already exists.")

    with open(out / "seed42_clipping_protocol.json") as f:
        protocol = json.load(f)

    with np.load(root / "data/galaxy_properties.npz") as data:
        names = list(data.files)
        Y = np.column_stack([data[k] for k in names]).astype(np.float64)

    limits = protocol["thresholds"]
    lower = np.array([limits[k]["lower"] for k in names])
    upper = np.array([limits[k]["upper"] for k in names])

    train, val = splits["train"], splits["val"]
    Ytr = np.clip(Y[train], lower, upper)
    Yva = np.clip(Y[val], lower, upper)
    device = config["device"]

    Ygpu = torch.tensor(Ytr, dtype=torch.float64, device=device)
    Ymean = Ygpu.mean(0)
    Ycenter = Ygpu - Ymean
    records = []

    for name in tqdm(model_names, desc="Individual teachers"):
        view = np.load(whitened_dir / f"{name}.npy", mmap_mode="r")
        assert view.shape[0] == config["num_galaxies"]

        X = torch.tensor(view[train], device=device, dtype=torch.float64)
        V = torch.tensor(view[val], device=device, dtype=torch.float64)

        mean = X.mean(0)
        scale = X.std(0, unbiased=False)
        scale = torch.where(scale > 0, scale, 1.0)

        X.sub_(mean).div_(scale)
        V.sub_(mean).div_(scale)

        gram = X.T @ X
        gram.diagonal().add_(100.0)
        weights = torch.linalg.solve(gram, X.T @ Ycenter)
        predicted = (V @ weights + Ymean).cpu().numpy()

        assert np.isfinite(predicted).all(), name

        if name == "convnext_nano":
            reference = make_pipeline(
                StandardScaler(),
                Ridge(alpha=100.0, solver="cholesky"),
            )
            reference.fit(view[train], Ytr)
            difference = np.max(np.abs(reference.predict(view[val]) - predicted))
            assert difference < 1e-3, f"Ridge parity failed: {difference}"
            print(f"\nGPU/sklearn parity max difference: {difference:.2e}")

        r2 = r2_score(Yva, predicted, multioutput="raw_values")
        mae = mean_absolute_error(Yva, predicted, multioutput="raw_values")

        records.extend(
            {
                "Teacher": name,
                "Dimensions": view.shape[1],
                "Property": prop,
                "R2": float(r2[j]),
                "MAE": float(mae[j]),
            }
            for j, prop in enumerate(names)
        )

        del X, V, gram, weights

    report = pd.DataFrame(records)

    existing = pd.read_csv(out / "seed42_validation_probes_clipped.csv")
    reference = existing.pivot(index="Property", columns="Source", values="R2")

    winners = (
        report.sort_values("R2", ascending=False)
        .drop_duplicates("Property")
        .set_index("Property")
        .loc[names]
    )

    comparison = pd.DataFrame(
        {
            "Student-22": reference.loc[names, "Student-22"],
            "Best teacher": winners["R2"],
            "Gestalt-22": reference.loc[names, "Gestalt-22"],
            "Best teacher name": winners["Teacher"],
        }
    )

    print("\nValidation R² Comparison: Student-22 vs. Best Individual Teachers vs. Gestalt-22 (Clipped Targets)\n")
    print(comparison.round(4).to_string())

    leaderboard = report.groupby("Teacher")["R2"].mean().sort_values(ascending=False)
    print("\nIndividual Teacher Ranking: Mean Validation R² Across 8 Galaxy Properties\n")
    print(leaderboard.round(4).to_string())

    fig, ax = plt.subplots(figsize=(9, 5))
    y = np.arange(len(names))
    for shift, column in zip(
        (-0.25, 0, 0.25),
        ("Student-22", "Best teacher", "Gestalt-22"),
    ):
        ax.barh(y + shift, comparison[column], height=0.24, label=column)

    ax.set_yticks(y, names)
    ax.invert_yaxis()
    ax.set_xlabel("Validation R² Comparison")
    ax.set_title("Student-22 vs individual-teacher best vs Gestalt-22")
    ax.legend()
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=160)
    plt.show()
    plt.close(fig)

    with open(result_path, "x") as f:
        report.to_csv(f, index=False)

    with open(method_path, "x") as f:
        json.dump(
            {
                "models": model_names,
                "split": "train → validation",
                "target_clipping": "train-only 1st/99th percentiles",
                "feature_scaling": "train-only z-score",
                "ridge_alpha": 100.0,
                "solver": "GPU float64 linear solve",
                "test_used": False,
            },
            f,
            indent=2,
        )

    print(f"\nEvaluated: {len(model_names)}/22 teachers")
    print("Test split: NOT ACCESSED")
    print("Saved:", result_path)
    print("Saved:", plot_path)

    return report
