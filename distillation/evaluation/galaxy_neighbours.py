import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


def compare_neighbors(
    config,
    splits,
    student_embeddings_dir,
    teacher_target_dir,
    pretrained_paths,
    frozen_paths,
):

    out = config["output_dir"] / "benchmarks"
    csv_path = out / "seed42_test_neighbour_recall.csv"
    fig_path = out / "seed42_test_neighbour_recall.png"

    if csv_path.exists() or fig_path.exists():
        raise FileExistsError("Neighbour benchmark artifacts already exist.")

    with np.load(teacher_target_dir / "target_normalization.npz") as norm:
        mean, std = norm["mean"], norm["std"]

    gestalt = np.load(teacher_target_dir / "gestalt_1024.npy", mmap_mode="r")

    reference = (np.asarray(gestalt[splits["test"]], dtype=np.float32) - mean) / std

    sources = {
        "Pretrained DINOv3": pretrained_paths["test"],
        "Frozen-distilled": frozen_paths["test"],
        "Student-12": student_embeddings_dir / "student_12_test.npy",
        "Student-22": student_embeddings_dir / "student_22_test.npy",
    }

    device = config["device"]
    n = len(splits["test"])
    assert reference.shape == (n, config["embedding_dim"])
    assert n > 50

    def neighbours(features, k=50, batch_size=256):
        x = torch.as_tensor(
            np.asarray(features, dtype=np.float32),
            device=device,
        )
        assert x.shape[0] == n
        assert torch.isfinite(x).all()
        assert (torch.linalg.vector_norm(x, dim=1) > 0).all()

        x = F.normalize(x, dim=1)
        indices = []

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            similarity = x[start:end] @ x.T

            rows = torch.arange(end - start, device=device)
            similarity[rows, rows + start] = -torch.inf

            nearest = similarity.topk(k, dim=1).indices
            indices.append(nearest.cpu().numpy())

        return np.concatenate(indices)

    teacher_nn = neighbours(reference)
    records = []

    for source, path in sources.items():
        features = np.load(path, mmap_mode="r")
        model_nn = neighbours(features)

        for k in (10, 50):
            matches = (
                (model_nn[:, :k, None] == teacher_nn[:, None, :k])
                .any(axis=2)
                .sum(axis=1)
            )

            records.append(
                {
                    "Representation": source,
                    "K": k,
                    "Recall": float(matches.mean() / k),
                    "Recall_std_across_queries": float((matches / k).std(ddof=1)),
                    "N_queries": n,
                    "Gallery_size": n,
                    "Metric": "cosine",
                }
            )

    report = pd.DataFrame(records)
    table = report.pivot(index="Representation", columns="K", values="Recall").loc[
        list(sources)
    ]

    print("\nTest Galaxy Neighbour Similarity\n")
    print(table.rename(columns={10: "Recall@10", 50: "Recall@50"}).round(4).to_string())

    ax = table.plot.barh(figsize=(8, 4), width=0.72)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Fraction of Gestalt neighbours recovered")
    ax.set_ylabel("")
    ax.set_title("Galaxy Neighbour Similarity Comparison")
    ax.legend(["Recall@10", "Recall@50"])
    ax.grid(axis="x", alpha=0.2)
    ax.figure.tight_layout()

    with csv_path.open("x") as f:
        report.to_csv(f, index=False)
    with fig_path.open("xb") as f:
        ax.figure.savefig(f, format="png", dpi=170)

    plt.show()
    plt.close(ax.figure)

    print("Saved:", csv_path)
    print("Saved:", fig_path)

    return report
