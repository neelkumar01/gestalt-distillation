import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def compare_quality_and_cost(config):

    bench = Path(config["output_dir"]) / "benchmarks"
    csv_path = bench / "seed42_exp1_point4_quality_cost.csv"
    fig_path = bench / "seed42_exp1_point4_quality_cost.png"

    if csv_path.exists() or fig_path.exists():
        raise FileExistsError("Point 4 outputs already exist.")

    families = {
        "ConvNeXt": ("convnext", 4),
        "ViT": ("vit", 3),
        "ViT-MAE": ("vitmae", 3),
        "AstroPT": ("astropt", 3),
        "I-JEPA": ("ijepa", 2),
        "V-JEPA": ("vjepa", 3),
        "CLIP": ("clip", 2),
        "LLaVA": ("llava", 2),
    }

    frames = []
    for family, (prefix, count) in families.items():
        df = pd.read_csv(bench / f"{prefix}_family_inference.csv")
        assert df["Model"].nunique() == count, f"{family}: model count mismatch"
        frames.append(df.assign(Family=family))

    student = pd.read_csv(bench / "seed42_student22_inference_matched.csv")
    assert set(student["Model"]) == {"Student-22"}
    frames.append(student.assign(Family="Student"))

    timing = pd.concat(frames, ignore_index=True)

    assert timing["Model"].nunique() == 23
    assert not timing.duplicated(["Model", "Batch"]).any()
    assert timing.groupby("Model")["Batch"].apply(lambda x: set(x) == {1, 64}).all()
    assert timing["Precision"].eq("BF16").all()
    assert timing["GPU"].nunique() == 1

    metrics = ["Median_ms", "P95_ms", "Throughput_images_s"]
    assert np.isfinite(timing[metrics + ["Parameters_M"]]).all().all()

    def key(name):
        text = re.sub(r"[^a-z0-9]", "", name.lower())
        return re.sub(r"^astropt0+", "astropt", text)

    quality = pd.read_csv(bench / "seed42_exp1_tuned_test_r2.csv")
    quality = quality[quality["Protocol"] == "Clipped"].copy()
    quality["Key"] = quality["Model"].map(key)
    scores = quality.groupby("Key")["Test_R2"].agg(["mean", "count"])

    required = {key(name) for name in timing["Model"]}
    assert required.issubset(scores.index)
    assert scores.loc[list(required), "count"].eq(8).all()

    wide = timing.pivot(index="Model", columns="Batch", values=metrics)
    wide.columns = [f"{metric}_B{batch}" for metric, batch in wide.columns]

    report = timing.groupby("Model")[["Family", "Parameters_M"]].first().join(wide)

    report["Mean_Test_R2"] = [scores.loc[key(name), "mean"] for name in report.index]
    report = report.sort_values("Mean_Test_R2", ascending=False)

    teacher = report[report["Family"] != "Student"]
    best_name = teacher["Mean_Test_R2"].idxmax()
    best = report.loc[best_name]
    stu = report.loc["Student-22"]

    print("=== EXP 1 · POINT 4 — QUALITY vs INFERENCE COST ===")
    print(f"GPU: {timing['GPU'].iloc[0]} | BF16")
    print("22 frozen teachers + Student-22 | Batch 1 & 64\n")

    columns = [
        "Family",
        "Mean_Test_R2",
        "Parameters_M",
        "Median_ms_B1",
        "P95_ms_B1",
        "Median_ms_B64",
        "Throughput_images_s_B64",
    ]
    print(report[columns].round(3).to_string())

    colors = plt.get_cmap("tab10").colors
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), layout="constrained")

    plots = [
        ("Median_ms_B1", "Batch-1 latency (ms) ↓"),
        ("Throughput_images_s_B64", "Batch-64 throughput (images/s) ↑"),
    ]

    for ax, (metric, xlabel) in zip(axes, plots):
        for i, family in enumerate(families):
            group = teacher[teacher["Family"] == family]
            ax.scatter(
                group[metric],
                group["Mean_Test_R2"],
                s=60,
                color=colors[i],
                alpha=0.85,
                label=family,
            )

        ax.scatter(
            stu[metric],
            stu["Mean_Test_R2"],
            s=240,
            marker="*",
            color="black",
            edgecolor="white",
            linewidth=0.7,
            zorder=5,
            label="Student-22",
        )

        ax.annotate(
            "Student-22",
            (stu[metric], stu["Mean_Test_R2"]),
            xytext=(7, 7),
            textcoords="offset points",
            fontsize=9,
            weight="bold",
        )
        ax.annotate(
            best_name.replace("_", " "),
            (best[metric], best["Mean_Test_R2"]),
            xytext=(7, -13),
            textcoords="offset points",
            fontsize=9,
        )

        ax.set_xscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Mean clipped test R² ↑")
        ax.grid(alpha=0.18)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        "Experiment 1 · Prediction quality vs inference cost\n"
        "Student-22: additional unlabeled distillation training · "
        "Teachers: frozen pretrained models",
        fontsize=12,
        weight="semibold",
    )
    fig.legend(
        *axes[0].get_legend_handles_labels(),
        loc="outside lower center",
        ncol=5,
        frameon=False,
    )

    with csv_path.open("x") as f:
        report.reset_index().to_csv(f, index=False)
    with fig_path.open("xb") as f:
        fig.savefig(f, format="png", dpi=180)

    plt.show()
    plt.close(fig)

    print(f"\nBest frozen teacher: {best_name}")
    print(f"Student − teacher ΔR²: {stu['Mean_Test_R2'] - best['Mean_Test_R2']:+.4f}")
    print(f"Batch-1 speedup: {best['Median_ms_B1'] / stu['Median_ms_B1']:.2f}×")
    print(
        "Batch-64 throughput ratio: "
        f"{stu['Throughput_images_s_B64'] / best['Throughput_images_s_B64']:.2f}×"
    )
    print("\nScope: GPU embedding forward; preprocessing excluded.")
    print("Architecture-specific input workloads; not equal-compute training.")
    print("Student-22 received extra unlabeled distillation training.")
    print("Individual teacher baselines were frozen pretrained models.")
    print(f"\nSaved: {csv_path}\nSaved: {fig_path}")

    return report


exp1_point4_report = report_exp1_point4(config)
