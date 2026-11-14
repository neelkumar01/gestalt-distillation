from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def summarize_best_teachers(config):

    bench = Path(config["output_dir"]) / "benchmarks"
    summary_path = bench / "seed42_exp1_best_teacher_comparison.csv"
    figure_path = bench / "seed42_exp1_best_teacher_ci.png"

    if summary_path.exists() or figure_path.exists():
        raise FileExistsError("Reporting artifacts already exist.")

    validation = pd.read_csv(bench / "seed42_exp1_ridge_validation.csv")
    results = pd.read_csv(bench / "seed42_exp1_paired_r2_ci.csv")

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
    protocols = ["Clipped", "Raw"]

    teacher_names = {
        p.stem
        for p in (Path(config["output_dir"]) / "teacher_embeddings").glob("*.npy")
    }
    assert len(teacher_names) == 22

    best = (
        validation[validation["Model"].isin(teacher_names)]
        .sort_values(
            ["Protocol", "Property", "Validation_R2", "Model"],
            ascending=[True, True, False, True],
        )
        .drop_duplicates(["Protocol", "Property"])[["Protocol", "Property", "Model"]]
    )

    summary = best.merge(
        results,
        on=["Protocol", "Property", "Model"],
        how="left",
        validate="one_to_one",
    ).rename(
        columns={
            "Model": "Best_Validation_Teacher",
            "Test_R2": "Teacher_Test_R2",
        }
    )

    assert len(summary) == 16
    assert (
        summary[["Teacher_Test_R2", "Student22_R2", "Delta_R2", "CI_Lower", "CI_Upper"]]
        .notna()
        .all()
        .all()
    )
    np.testing.assert_allclose(
        summary["Delta_R2"],
        summary["Student22_R2"] - summary["Teacher_Test_R2"],
        atol=1e-8,
    )

    summary["Protocol"] = pd.Categorical(
        summary["Protocol"], categories=protocols, ordered=True
    )
    summary["Property"] = pd.Categorical(
        summary["Property"], categories=properties, ordered=True
    )
    summary = summary.sort_values(["Protocol", "Property"])

    clipped = summary[summary["Protocol"] == "Clipped"].copy()

    print("Final Results Summary (Seed 42)")
    print("Comparator: strongest individual teacher per property,")
    print("selected using validation R² only.\n")

    display_cols = [
        "Property",
        "Best_Validation_Teacher",
        "Student22_R2",
        "Teacher_Test_R2",
        "Delta_R2",
        "CI_Lower",
        "CI_Upper",
    ]
    print(clipped[display_cols].round(4).to_string(index=False))

    positive = (clipped["CI_Lower"] > 0).sum()
    negative = (clipped["CI_Upper"] < 0).sum()
    print(
        f"\nClipped: {positive}/8 positive CIs, "
        f"{negative}/8 negative CIs, "
        f"{8 - positive - negative}/8 include zero."
    )

    y = np.arange(len(clipped))
    delta = clipped["Delta_R2"].to_numpy()
    errors = np.vstack(
        [
            delta - clipped["CI_Lower"].to_numpy(),
            clipped["CI_Upper"].to_numpy() - delta,
        ]
    )

    fig, ax = plt.subplots(figsize=(9, 4.8), layout="constrained")
    ax.errorbar(
        delta,
        y,
        xerr=errors,
        fmt="o",
        capsize=3,
        color="tab:blue",
        ecolor="tab:gray",
    )
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_yticks(y, clipped["Property"])
    ax.invert_yaxis()
    ax.set_xlabel("Student-22 − best validation teacher (test R²)")
    ax.set_title("Galaxy Property Prediction Comparison with 95% Confidence Intervals")
    ax.grid(axis="x", alpha=0.2)

    with summary_path.open("x") as file:
        summary.to_csv(file, index=False)
    with figure_path.open("xb") as file:
        fig.savefig(file, format="png", dpi=180)

    plt.show()
    plt.close(fig)

    print("\nSaved:", summary_path)
    print("Saved:", figure_path)
    print("All 27-model results remain in seed42_exp1_paired_r2_ci.csv.")

    return summary
