import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


def compare_geometry(
    config,
    splits,
    teacher_target_dir,
    student_embeddings_dir,
    pretrained_paths,
    frozen_paths,
):

    out = config["output_dir"] / "benchmarks"
    csv_path = out / "seed42_global_geometry.csv"
    fig_path = out / "seed42_global_geometry.png"
    if csv_path.exists() or fig_path.exists():
        raise FileExistsError("Geometry artifacts already exist.")

    device = config["device"]

    with np.load(teacher_target_dir / "target_normalization.npz") as z:
        mean, std = z["mean"], z["std"]

    gestalt = np.load(teacher_target_dir / "gestalt_1024.npy", mmap_mode="r")
    target = (gestalt[splits["test"]].astype(np.float32) - mean) / std

    paths = {
        "Pretrained DINOv3": pretrained_paths["test"],
        "Frozen-distilled": frozen_paths["test"],
        "Student-12": student_embeddings_dir / "student_12_test.npy",
        "Student-22": student_embeddings_dir / "student_22_test.npy",
    }

    n = len(splits["test"])
    assert target.shape == (n, config["embedding_dim"])

    rows, cols = torch.triu_indices(n, n, offset=1, device=device)
    n_pairs = len(rows)

    def geometry(features):
        x = torch.as_tensor(np.asarray(features, dtype=np.float32), device=device)
        assert x.shape[0] == n and torch.isfinite(x).all()
        assert (torch.linalg.vector_norm(x, dim=1) > 0).all()

        unit = F.normalize(x, dim=1)
        similarities = (unit @ unit.T)[rows, cols]

        centered = x - x.mean(dim=0, keepdim=True)
        eigenvalues = torch.linalg.eigvalsh(centered.T @ centered)
        eigenvalues = eigenvalues.clamp_min(0)
        p = eigenvalues / eigenvalues.sum()
        positive = p[p > 0]
        effective_rank = torch.exp(-(positive * positive.log()).sum()).item()

        descending = torch.flip(p, dims=[0])
        rank90 = int(
            torch.searchsorted(
                descending.cumsum(0).contiguous(), torch.tensor(0.9, device=device)
            ).item()
            + 1
        )

        return similarities, effective_rank, rank90

    reference, teacher_rank, teacher_rank90 = geometry(target)
    ref = reference.double()
    ref_centered = ref - ref.mean()
    ref_norm = torch.linalg.vector_norm(ref_centered)

    records = [
        {
            "Representation": "Gestalt-22",
            "Pair_Cosine_Pearson": 1.0,
            "Pair_Cosine_MAE": 0.0,
            "Effective_Rank": teacher_rank,
            "Rank90": teacher_rank90,
            "N_pairs": n_pairs,
            "N_test": n,
        }
    ]
    student22_pairs = None

    for source, path in paths.items():
        similarities, rank, rank90 = geometry(np.load(path, mmap_mode="r"))
        value = similarities.double()
        shifted = value - value.mean()

        correlation = (
            torch.dot(ref_centered, shifted)
            / (ref_norm * torch.linalg.vector_norm(shifted))
        ).item()
        mae = torch.mean(torch.abs(ref - value)).item()

        assert np.isfinite([correlation, mae, rank]).all()

        records.append(
            {
                "Representation": source,
                "Pair_Cosine_Pearson": correlation,
                "Pair_Cosine_MAE": mae,
                "Effective_Rank": rank,
                "Rank90": rank90,
                "N_pairs": n_pairs,
                "N_test": n,
            }
        )

        if source == "Student-22":
            student22_pairs = similarities.cpu().numpy()

    report = pd.DataFrame(records)

    print("\nGalaxy Similarity Structure Comparison on Test Set\n")
    print(report.drop(columns=["N_pairs", "N_test"]).round(4).to_string(index=False))
    print(f"\nGalaxies: {n:,} | Unique non-self pairs: {n_pairs:,}")
    print("Pearson: higher better | MAE: lower better")
    print("Effective rank / Rank90: compare against Gestalt, not maximize blindly")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")

    compared = report.iloc[1:]
    axes[0].barh(
        compared["Representation"],
        compared["Pair_Cosine_Pearson"],
    )
    axes[0].invert_yaxis()
    axes[0].set_xlim(0, 1)
    axes[0].set_xlabel("Pearson correlation")
    axes[0].set_title("Galaxy Pair Similarity Agreement with Gestalt-22")
    axes[0].grid(axis="x", alpha=0.2)

    rng = np.random.default_rng(42)
    chosen = rng.choice(n_pairs, size=min(50000, n_pairs), replace=False)
    x_plot = reference.cpu().numpy()[chosen]
    y_plot = student22_pairs[chosen]

    density = axes[1].hexbin(
        x_plot,
        y_plot,
        gridsize=55,
        bins="log",
        mincnt=1,
        cmap="viridis",
    )
    bounds = [
        min(x_plot.min(), y_plot.min()),
        max(x_plot.max(), y_plot.max()),
    ]
    axes[1].plot(bounds, bounds, "--", color="gray", linewidth=1)
    axes[1].set_xlabel("Gestalt pairwise cosine")
    axes[1].set_ylabel("Student-22 pairwise cosine")
    axes[1].set_title("Student-22 vs Gestalt-22 Galaxy Pair Similarity")
    fig.colorbar(density, ax=axes[1], label="Pair density (log)")

    with csv_path.open("x") as f:
        report.to_csv(f, index=False)
    with fig_path.open("xb") as f:
        fig.savefig(f, format="png", dpi=170)

    plt.show()
    plt.close(fig)

    print("\nSaved:", csv_path)
    print("Saved:", fig_path)

    return report
