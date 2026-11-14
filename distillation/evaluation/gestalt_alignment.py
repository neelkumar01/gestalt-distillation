import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt


def compare_with_gestalt(
    config,
    splits,
    student_embeddings_dir,
    teacher_target_dir,
    pretrained_paths,
    frozen_paths,
):

    out = config["output_dir"] / "benchmarks"
    csv_path = out / "seed42_test_representation_fidelity.csv"
    fig_path = out / "seed42_test_representation_fidelity.png"

    if csv_path.exists() or fig_path.exists():
        raise FileExistsError("Fidelity artifacts already exist.")

    with np.load(teacher_target_dir / "target_normalization.npz") as norm:
        mean = norm["mean"]
        std = norm["std"]

    gestalt = np.load(teacher_target_dir / "gestalt_1024.npy", mmap_mode="r")

    target = (np.asarray(gestalt[splits["test"]], dtype=np.float64) - mean) / std

    assert target.shape == (len(splits["test"]), config["embedding_dim"])
    assert np.isfinite(target).all()

    paths = {
        "Pretrained DINOv3": pretrained_paths["test"],
        "Frozen-distilled": frozen_paths["test"],
        "Student-12": student_embeddings_dir / "student_12_test.npy",
        "Student-22": student_embeddings_dir / "student_22_test.npy",
    }

    device = config["device"]
    teacher = torch.as_tensor(target, dtype=torch.float64, device=device)

    def centered_gram(x):
        x = x - x.mean(dim=0, keepdim=True)
        return x.T @ x

    teacher_centered = teacher - teacher.mean(0, keepdim=True)
    teacher_gram = centered_gram(teacher)
    teacher_norm = torch.linalg.vector_norm(teacher_gram)
    teacher_variance = teacher_centered.square().sum()

    records = []

    for source, path in paths.items():
        features = np.load(path, mmap_mode="r")
        assert features.shape[0] == len(splits["test"])
        assert np.isfinite(features).all(), source

        x = torch.as_tensor(
            np.asarray(features),
            dtype=torch.float64,
            device=device,
        )

        centered = x - x.mean(0, keepdim=True)
        cross = centered.T @ teacher_centered

        cka = (
            cross.square().sum()
            / (torch.linalg.vector_norm(centered.T @ centered) * teacher_norm)
        ).item()

        row = {
            "Representation": source,
            "Dimensions": features.shape[1],
            "Normalized_MSE": np.nan,
            "Embedding_R2": np.nan,
            "Mean_Cosine": np.nan,
            "Linear_CKA": float(cka),
            "N_test": len(splits["test"]),
        }

        if x.shape == teacher.shape:
            error = (x - teacher).square()
            mse = error.mean().item()
            r2 = (1 - error.sum() / teacher_variance).item()
            cosine = (
                torch.nn.functional.cosine_similarity(x, teacher, dim=1, eps=1e-12)
                .mean()
                .item()
            )

            row.update(
                {
                    "Normalized_MSE": mse,
                    "Embedding_R2": r2,
                    "Mean_Cosine": cosine,
                }
            )

        records.append(row)
        del x, centered, cross

    report = pd.DataFrame(records)

    assert np.isfinite(report["Linear_CKA"]).all()
    assert report["Linear_CKA"].between(0, 1.00001).all()
    assert np.isfinite(
        report.loc[
            report["Representation"] != "Pretrained DINOv3",
            ["Normalized_MSE", "Embedding_R2", "Mean_Cosine"],
        ].to_numpy()
    ).all()

    print("\nTest Embedding Similarity with Gestalt-22\n")
    print(report.round(4).to_string(index=False))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(
        report["Representation"],
        report["Linear_CKA"],
        color=plt.cm.Blues(np.linspace(0.4, 0.85, len(report))),
    )
    ax.set_xlim(0, 1)
    ax.invert_yaxis()
    ax.set_xlabel("Linear CKA with Gestalt-22")
    ax.set_title("Embedding Similarity on Test Galaxies")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()

    with csv_path.open("x") as f:
        report.to_csv(f, index=False)
    with fig_path.open("xb") as f:
        fig.savefig(f, format="png", dpi=170)

    plt.show()
    plt.close(fig)

    print("\nSaved:", csv_path)
    print("Saved:", fig_path)

    return report
