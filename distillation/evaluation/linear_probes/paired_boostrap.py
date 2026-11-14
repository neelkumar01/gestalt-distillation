import numpy as np
import pandas as pd
from pathlib import Path


def bootstrap_r2_differences(config, n_boot=2000, seed=42):

    bench = Path(config["output_dir"]) / "benchmarks"
    output = bench / "seed42_exp1_paired_r2_ci.csv"
    if output.exists():
        raise FileExistsError(f"Will not overwrite: {output}")

    with np.load(bench / "seed42_exp1_tuned_predictions.npz") as data:
        models = data["models"].tolist()
        protocols = data["protocols"].tolist()
        properties = data["properties"].tolist()
        ids = data["test_indices"]
        predictions = data["predictions"].astype(np.float64)
        truth = data["y_true"].astype(np.float64)

    m, q, n, p = predictions.shape
    assert (m, q, n, p) == (27, 2, 4500, 8)
    assert truth.shape == (q, n, p)
    assert len(np.unique(ids)) == n
    assert len(set(models)) == m
    assert np.isfinite(predictions).all() and np.isfinite(truth).all()

    student_idx = models.index("Student-22")

    y = truth.transpose(1, 0, 2).reshape(n, q * p)
    pred = predictions.transpose(2, 0, 1, 3).reshape(n, m, q * p)

    squared_error = (pred - y[:, None, :]) ** 2
    tss = ((y - y.mean(axis=0)) ** 2).sum(axis=0)
    assert np.all(tss > 0)

    r2 = 1 - squared_error.sum(axis=0) / tss
    delta = r2[student_idx] - r2

    previous = pd.read_csv(bench / "seed42_exp1_tuned_test_r2.csv")
    keys = pd.MultiIndex.from_product(
        [models, protocols, properties],
        names=["Model", "Protocol", "Property"],
    )
    assert len(previous) == len(keys)
    assert not previous.duplicated(list(keys.names)).any()

    saved = previous.set_index(list(keys.names))["Test_R2"].reindex(keys)
    assert saved.notna().all()
    np.testing.assert_allclose(r2.reshape(-1), saved.to_numpy(), rtol=0, atol=1e-4)

    differences = (
        squared_error - squared_error[:, student_idx : student_idx + 1, :]
    ).reshape(n, m * q * p)

    rng = np.random.default_rng(seed)
    bootstrap = np.empty((n_boot, m, q * p), dtype=np.float64)

    for start in range(0, n_boot, 100):
        count = min(100, n_boot - start)
        draws = rng.integers(n, size=(count, n))
        offsets = n * np.arange(count)[:, None]
        weights = np.bincount(
            (draws + offsets).ravel(),
            minlength=count * n,
        ).reshape(count, n)

        resampled_tss = weights @ (y**2) - (weights @ y) ** 2 / n
        assert np.all(resampled_tss > 0)

        bootstrap[start : start + count] = (weights @ differences).reshape(
            count, m, q * p
        ) / resampled_tss[:, None, :]

    lower, upper = np.percentile(bootstrap, [2.5, 97.5], axis=0)

    rows = []
    for i, model in enumerate(models):
        for j, (protocol, prop) in enumerate(
            (protocol, prop) for protocol in protocols for prop in properties
        ):
            rows.append(
                {
                    "Model": model,
                    "Protocol": protocol,
                    "Property": prop,
                    "Test_R2": r2[i, j],
                    "Student22_R2": r2[student_idx, j],
                    "Delta_R2": delta[i, j],
                    "CI_Lower": lower[i, j],
                    "CI_Upper": upper[i, j],
                }
            )

    report = pd.DataFrame(rows)
    with output.open("x") as file:
        report.to_csv(file, index=False)

    validation = pd.read_csv(bench / "seed42_exp1_ridge_validation.csv")
    teacher_names = {
        path.stem
        for path in (Path(config["output_dir"]) / "teacher_embeddings").glob("*.npy")
    }
    best_teacher = (
        validation[
            (validation["Protocol"] == "Clipped")
            & validation["Model"].isin(teacher_names)
        ]
        .groupby("Model")["Validation_R2"]
        .mean()
        .idxmax()
    )

    preview = report[
        (report["Model"] == best_teacher) & (report["Protocol"] == "Clipped")
    ][["Property", "Delta_R2", "CI_Lower", "CI_Upper"]].copy()

    print(f"Models: {m} | Properties: {p} | Protocols: {q}")
    print(f"Test galaxies: {n:,} | Paired bootstrap resamples: {n_boot:,}")
    print(f"\nStudent-22 minus {best_teacher} (validation-selected teacher):")
    print(preview.round(4).to_string(index=False))
    print("\nPositive ΔR² favors Student-22.")
    print(f"\nSaved: {output}")

    return report
