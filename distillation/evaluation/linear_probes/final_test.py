import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from sklearn.metrics import r2_score, mean_absolute_error


def final_heldout_benchmark(
    config,
    splits,
    student_embeddings_dir,
    teacher_target_dir,
    whitened_dir,
    model_names,
    selected_teachers_path,
):
    root = config["output_dir"]
    out = root / "benchmarks"
    csv = out / "final_test_all_representations.csv"
    figure = out / "final_test_clipped_comparison.png"

    if csv.exists() or figure.exists():
        raise FileExistsError("Final test artifacts already exist.")

    winners = json.loads(selected_teachers_path.read_text())
    with open(out / "seed42_clipping_protocol.json") as f:
        thresholds = json.load(f)["thresholds"]

    with np.load(root / "data/galaxy_properties.npz") as data:
        names = list(data.files)
        Y = np.column_stack([data[k] for k in names]).astype(np.float64)

    assert len(names) == len(winners) == 8
    assert set(names) == set(winners)

    train, test = splits["train"], splits["test"]
    lower = np.array([thresholds[k]["lower"] for k in names])
    upper = np.array([thresholds[k]["upper"] for k in names])

    Ytrain = np.concatenate(
        [
            Y[train],
            np.clip(Y[train], lower, upper),
        ],
        axis=1,
    )
    Ytest = {
        "raw": Y[test],
        "clipped": np.clip(Y[test], lower, upper),
    }

    device = config["device"]
    T = torch.tensor(Ytrain, dtype=torch.float64, device=device)
    target_mean = T.mean(0)
    centered_targets = T - target_mean
    records = []

    gestalt = np.load(teacher_target_dir / "gestalt_1024.npy", mmap_mode="r")

    sources = ["Student-12", "Student-22", "Gestalt-22"] + model_names
    assert len(sources) == 25

    for source in tqdm(sources, desc="Held-out probes"):
        if source.startswith("Student-"):
            stage = source.split("-")[1]
            Xtrain = np.load(student_embeddings_dir / f"student_{stage}_train.npy")
            Xtest = np.load(student_embeddings_dir / f"student_{stage}_test.npy")
        elif source == "Gestalt-22":
            Xtrain, Xtest = gestalt[train], gestalt[test]
        else:
            view = np.load(whitened_dir / f"{source}.npy", mmap_mode="r")
            Xtrain, Xtest = view[train], view[test]

        assert Xtrain.shape[0] == len(train)
        assert Xtest.shape == (len(test), Xtrain.shape[1])

        X = torch.tensor(Xtrain, dtype=torch.float64, device=device)
        V = torch.tensor(Xtest, dtype=torch.float64, device=device)
        mean = X.mean(0)
        scale = X.std(0, unbiased=False).clamp_min(1e-12)
        X = (X - mean) / scale
        V = (V - mean) / scale

        gram = X.T @ X
        gram.diagonal().add_(100.0)
        weights = torch.linalg.solve(gram, X.T @ centered_targets)
        prediction = (V @ weights + target_mean).cpu().numpy()

        assert np.isfinite(prediction).all(), source
        dim = Xtrain.shape[1]

        for protocol, start in (("raw", 0), ("clipped", len(names))):
            truth = Ytest[protocol]
            pred = prediction[:, start : start + len(names)]

            r2 = r2_score(truth, pred, multioutput="raw_values")
            mae = mean_absolute_error(truth, pred, multioutput="raw_values")

            for j, property_name in enumerate(names):
                records.append(
                    {
                        "Protocol": protocol,
                        "Source": source,
                        "Property": property_name,
                        "Dimensions": dim,
                        "R2": float(r2[j]),
                        "MAE": float(mae[j]),
                        "N_train": len(train),
                        "N_test": len(test),
                    }
                )

        del X, V, gram, weights

    report = pd.DataFrame(records)
    assert len(report) == 25 * 8 * 2

    for protocol in ("raw", "clipped"):
        pivot = report[report["Protocol"] == protocol].pivot(
            index="Property", columns="Source", values="R2"
        )

        summary = pivot.loc[names, ["Student-12", "Student-22", "Gestalt-22"]].copy()
        summary["Selected teacher"] = [pivot.loc[name, winners[name]] for name in names]
        summary["Δ 22−12"] = summary["Student-22"] - summary["Student-12"]

        print(f"\n{protocol.upper()} TARGETS")
        print(summary.round(4).to_string())

        if protocol == "clipped":
            clipped_summary = summary

    fig, ax = plt.subplots(figsize=(9, 4.8))
    columns = ["Student-12", "Student-22", "Selected teacher", "Gestalt-22"]
    y = np.arange(len(names))

    for i, column in enumerate(columns):
        ax.barh(
            y + (i - 1.5) * 0.19,
            clipped_summary[column],
            height=0.18,
            label=column,
        )

    ax.set_yticks(y, names)
    ax.invert_yaxis()
    ax.set_xlabel("Test R² (Clipped Targets)")
    ax.set_title("Final Test R² Comparison of All Models")
    ax.grid(axis="x", alpha=0.2)
    ax.legend(fontsize=8)
    fig.tight_layout()

    with csv.open("x") as f:
        report.to_csv(f, index=False)
    with figure.open("xb") as f:
        fig.savefig(f, format="png", dpi=160)

    plt.show()
    plt.close(fig)

    print("Saved:", csv)
    print("Saved:", figure)

    return report
