import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def measure_gain_recovery(config):

    out = config["output_dir"] / "benchmarks"
    csv_path = out / "seed42_gestalt_gain_recovery.csv"
    fig_path = out / "seed42_gestalt_gain_recovery.png"

    if csv_path.exists() or fig_path.exists():
        raise FileExistsError("Gain-recovery artifacts already exist.")

    original = pd.read_csv(out / "final_test_all_representations.csv")
    baselines = pd.read_csv(out / "pretrained_frozen_test_probes.csv")

    data = pd.concat([original, baselines], ignore_index=True)
    sources = [
        "Pretrained DINOv3",
        "Student-12",
        "Student-22",
        "Gestalt-22",
    ]
    protocols = ("raw", "clipped")
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

    data = data[data["Source"].isin(sources) & data["Protocol"].isin(protocols)]
    assert not data.duplicated(["Protocol", "Source", "Property"]).any()

    r2 = data.pivot(
        index=["Protocol", "Property"],
        columns="Source",
        values="R2",
    ).reindex(
        pd.MultiIndex.from_product(
            [protocols, properties],
            names=["Protocol", "Property"],
        )
    )

    assert r2[sources].notna().all().all()
    assert np.isfinite(r2[sources].to_numpy()).all()

    gap = r2["Gestalt-22"] - r2["Pretrained DINOv3"]
    if (gap <= 1e-6).any():
        raise ValueError("Gestalt–Pretrained R² gap is nonpositive or too small.")

    records = []
    for stage in ("Student-12", "Student-22"):
        recovered = r2[stage] - r2["Pretrained DINOv3"]

        for (protocol, prop), gain in recovered.items():
            records.append(
                {
                    "Protocol": protocol,
                    "Property": prop,
                    "Student": stage,
                    "R2_Pretrained": r2.loc[(protocol, prop), "Pretrained DINOv3"],
                    "R2_Student": r2.loc[(protocol, prop), stage],
                    "R2_Gestalt": r2.loc[(protocol, prop), "Gestalt-22"],
                    "Gestalt_Gain_R2": gap.loc[(protocol, prop)],
                    "Student_Gain_R2": gain,
                    "Gain_Recovered_pct": 100 * gain / gap.loc[(protocol, prop)],
                }
            )

    report = pd.DataFrame(records)

    print("\nGESTALT ADDITIONAL R² GAIN RECOVERED\n")
    print("Definition: 100 × (Student − Pretrained) / (Gestalt − Pretrained)")

    for protocol in protocols:
        subset = report[report["Protocol"] == protocol]
        table = subset.pivot(
            index="Property",
            columns="Student",
            values="Gain_Recovered_pct",
        ).loc[properties, ["Student-12", "Student-22"]]

        print(f"\n{protocol.upper()} TARGETS")
        print(table.round(1).to_string())
        print("Macro-average:", table.mean().round(1).to_dict())
        print(
            "Smallest Gestalt–Pretrained R² gap:",
            f"{gap.loc[protocol].min():.4f}",
        )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 5.2),
        sharey=True,
        layout="constrained",
    )
    y = np.arange(len(properties))
    height = 0.35

    for ax, protocol in zip(axes, protocols):
        subset = report[report["Protocol"] == protocol]
        table = subset.pivot(
            index="Property",
            columns="Student",
            values="Gain_Recovered_pct",
        ).loc[properties]

        ax.barh(
            y - height / 2,
            table["Student-12"],
            height,
            label="Student-12",
        )
        ax.barh(
            y + height / 2,
            table["Student-22"],
            height,
            label="Student-22",
        )
        ax.axvline(
            100,
            color="gray",
            linestyle="--",
            linewidth=1,
            label="Gestalt (100%)",
        )
        ax.set_title(protocol.capitalize())
        ax.set_xlabel("Additional R² gain recovered (%)")
        ax.set_yticks(y, properties)
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.2)
        ax.set_axisbelow(True)

    axes[1].legend(loc="lower right", fontsize=8)
    fig.suptitle("Gestalt gain recovered beyond pretrained DINOv3")

    with csv_path.open("x") as f:
        report.to_csv(f, index=False)
    with fig_path.open("xb") as f:
        fig.savefig(f, format="png", dpi=180)

    plt.show()
    plt.close(fig)

    print("\nSaved:", csv_path)
    print("Saved:", fig_path)

    return report
