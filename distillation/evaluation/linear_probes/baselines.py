import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error


def evaluate_backbone_baselines(config, splits, pretrained_paths, frozen_paths):

    root = config["output_dir"]
    out = root / "benchmarks"
    csv_path = out / "pretrained_frozen_test_probes.csv"
    fig_path = out / "five_representation_clipped_test_r2.png"

    if csv_path.exists() or fig_path.exists():
        raise FileExistsError("Baseline benchmark artifacts already exist.")

    with np.load(root / "data/galaxy_properties.npz") as data:
        names = list(data.files)
        Y = np.column_stack([data[k] for k in names]).astype(np.float64)

    with open(out / "seed42_clipping_protocol.json") as f:
        thresholds = json.load(f)["thresholds"]

    lo = np.array([thresholds[k]["lower"] for k in names])
    hi = np.array([thresholds[k]["upper"] for k in names])

    train, test = splits["train"], splits["test"]
    Y_train = np.column_stack(
        [
            Y[train],
            np.clip(Y[train], lo, hi),
        ]
    )

    truths = {
        "raw": Y[test],
        "clipped": np.clip(Y[test], lo, hi),
    }
    assert np.isfinite(Y_train).all()
    assert all(np.isfinite(y).all() for y in truths.values())

    results = []

    for source, paths in (
        ("Pretrained DINOv3", pretrained_paths),
        ("Frozen-distilled", frozen_paths),
    ):
        X_train = np.load(paths["train"], mmap_mode="r")
        X_test = np.load(paths["test"], mmap_mode="r")

        assert X_train.shape[0] == len(train)
        assert X_test.shape == (len(test), X_train.shape[1])

        probe = make_pipeline(
            StandardScaler(),
            Ridge(alpha=100.0, solver="cholesky"),
        )
        probe.fit(X_train, Y_train)
        predictions = probe.predict(X_test)

        assert np.isfinite(predictions).all(), source

        for protocol, offset in (("raw", 0), ("clipped", len(names))):
            actual = truths[protocol]
            predicted = predictions[:, offset : offset + len(names)]

            r2 = r2_score(actual, predicted, multioutput="raw_values")
            mae = mean_absolute_error(actual, predicted, multioutput="raw_values")

            for j, name in enumerate(names):
                results.append(
                    {
                        "Protocol": protocol,
                        "Source": source,
                        "Property": name,
                        "Dimensions": X_train.shape[1],
                        "R2": float(r2[j]),
                        "MAE": float(mae[j]),
                        "N_train": len(train),
                        "N_test": len(test),
                    }
                )

    report = pd.DataFrame(results)
    assert len(report) == 2 * 2 * len(names)

    existing = pd.read_csv(out / "final_test_all_representations.csv")
    selected = existing[
        existing["Source"].isin(["Student-12", "Student-22", "Gestalt-22"])
    ]

    combined = pd.concat([report, selected], ignore_index=True)
    sources = [
        "Pretrained DINOv3",
        "Frozen-distilled",
        "Student-12",
        "Student-22",
        "Gestalt-22",
    ]

    print("\nFinal Test R² Comparison\n")

    for protocol in ("raw", "clipped"):
        matrix = (
            combined[combined["Protocol"] == protocol]
            .pivot(index="Property", columns="Source", values="R2")
            .loc[names, sources]
        )

        assert np.isfinite(matrix.to_numpy()).all()

        print(f"\n{protocol.upper()} TARGETS")
        print(matrix.round(4).to_string())

        if protocol == "clipped":
            clipped = matrix

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(
        clipped.to_numpy(),
        aspect="auto",
        cmap="viridis",
        vmin=0,
        vmax=1,
    )
    ax.set_xticks(range(len(sources)), sources, rotation=20, ha="right")
    ax.set_yticks(range(len(names)), names)

    for i in range(len(names)):
        for j in range(len(sources)):
            ax.text(
                j,
                i,
                f"{clipped.iloc[i, j]:.3f}",
                ha="center",
                va="center",
                fontsize=9,
                color="white" if clipped.iloc[i, j] < 0.55 else "black",
            )

    ax.set_title("Test R² Comparison of Five Models (Clipped Targets)")
    fig.colorbar(im, ax=ax, label="R²", shrink=0.8)
    fig.tight_layout()

    with csv_path.open("x") as f:
        report.to_csv(f, index=False)

    with fig_path.open("xb") as f:
        fig.savefig(f, format="png", dpi=170)

    plt.show()
    plt.close(fig)

    print("\nSaved:", csv_path)
    print("Saved:", fig_path)
    print("Ridge α: 100")

    return report
