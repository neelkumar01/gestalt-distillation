import gc
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from transformers import AutoVideoProcessor, AutoModel


def benchmark_vjepa(config, splits):

    models = {
        "vjepa_large": "facebook/vjepa2-vitl-fpc64-256",
        "vjepa_huge": "facebook/vjepa2-vith-fpc64-256",
        "vjepa_giant": "facebook/vjepa2-vitg-fpc64-256",
    }

    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    bench_dir = Path(config["output_dir"]) / "benchmarks"
    csv_path = bench_dir / "vjepa_family_inference.csv"
    plot_path = bench_dir / "vjepa_family_quality_cost.png"

    if csv_path.exists() or plot_path.exists():
        raise FileExistsError("V-JEPA family outputs already exist.")

    quality = pd.read_csv(bench_dir / "seed42_exp1_tuned_test_r2.csv")
    quality = quality[
        quality["Protocol"].str.lower().eq("clipped") & quality["Model"].isin(models)
    ]

    scores = quality.groupby("Model")["Test_R2"].agg(["mean", "count"])
    assert set(scores.index) == set(models)
    assert scores["count"].eq(8).all(), "Expected 8 properties per model."

    images = np.load(
        Path(config["output_dir"]) / "data" / "hsc_images.npy",
        mmap_mode="r",
    )
    image = np.asarray(images[splits["test"][0]])

    def measure(model, clip, batch):
        x = clip.repeat(batch, 1, 1, 1, 1)

        def forward():
            return model(
                pixel_values_videos=x,
                skip_predictor=True,
            ).last_hidden_state.mean(dim=1)

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for _ in range(10):
                forward()

            torch.cuda.synchronize()
            times = []

            for _ in range(40):
                torch.cuda.synchronize()
                start = time.perf_counter()

                forward()

                torch.cuda.synchronize()
                times.append((time.perf_counter() - start) * 1000)

        median = float(np.median(times))
        p95 = float(np.percentile(times, 95))
        return median, p95, batch * 1000 / median

    gpu = torch.cuda.get_device_name(0)
    rows = []

    print("=== V-JEPA FAMILY · INFERENCE BENCHMARK ===")
    print(f"GPU: {gpu}")
    print("BF16 | 10 warmups | 40 timed runs | Batch 1 & 64")
    print("16 repeated frames | Encoder only | Mean-token pooling\n")

    for name, checkpoint in models.items():
        print(f"Benchmarking {name} ...")

        processor = AutoVideoProcessor.from_pretrained(checkpoint)
        model = AutoModel.from_pretrained(checkpoint).to("cuda").eval()

        frame = processor(image, return_tensors="pt")["pixel_values_videos"]

        assert frame.ndim == 5 and frame.shape[:3] == (1, 1, 3)
        clip = frame.repeat(1, 16, 1, 1, 1).to("cuda")

        params = sum(p.numel() for p in model.parameters()) / 1e6
        encoder_params = sum(p.numel() for p in model.encoder.parameters()) / 1e6

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            embedding = model(
                pixel_values_videos=clip,
                skip_predictor=True,
            ).last_hidden_state.mean(dim=1)

        cached = np.load(
            Path(config["output_dir"]) / "teacher_embeddings" / f"{name}.npy",
            mmap_mode="r",
        )

        assert cached.shape == (
            len(images),
            embedding.shape[1],
        ), f"Cached embedding dimension mismatch: {name}"

        print(f"  Parameters: {params:.2f}M total, " f"{encoder_params:.2f}M encoder")
        print(
            f"  Input: {clip.shape[1]} frames × "
            f"{clip.shape[-2]}×{clip.shape[-1]} | "
            f"Embedding: {embedding.shape[1]}"
        )

        del embedding, cached

        for batch in (1, 64):
            try:
                median, p95, throughput = measure(model, clip, batch)
                status = "OK"

            except torch.cuda.OutOfMemoryError:
                if batch == 1:
                    raise

                median = p95 = throughput = float("nan")
                status = "OOM"
                gc.collect()
                torch.cuda.empty_cache()

            rows.append(
                {
                    "Model": name,
                    "Checkpoint": checkpoint,
                    "Batch": batch,
                    "Median_ms": median,
                    "P95_ms": p95,
                    "Throughput_images_s": throughput,
                    "Parameters_M": params,
                    "Encoder_Parameters_M": encoder_params,
                    "Frames": 16,
                    "Input_Size": clip.shape[-1],
                    "Precision": "BF16",
                    "GPU": gpu,
                    "Status": status,
                }
            )

            if status == "OK":
                print(
                    f"  Batch {batch:>2}: {median:.3f} ms | "
                    f"P95 {p95:.3f} ms | "
                    f"{throughput:,.1f} images/s"
                )
            else:
                print(f"  Batch {batch:>2}: OOM — not measured")

        del model, processor, frame, clip
        gc.collect()
        torch.cuda.empty_cache()
        print()

    results = pd.DataFrame(rows)

    summary = results.pivot(
        index="Model",
        columns="Batch",
        values=["Median_ms", "P95_ms", "Throughput_images_s"],
    )
    summary.columns = [f"{metric}_B{batch}" for metric, batch in summary.columns]

    summary = summary.join(
        results.groupby("Model")[["Parameters_M", "Encoder_Parameters_M"]].first()
    ).reindex(models)

    summary["Mean_Test_R2"] = scores.loc[summary.index, "mean"]

    print("=== FINAL V-JEPA FAMILY COMPARISON ===")
    print(
        summary[
            [
                "Parameters_M",
                "Encoder_Parameters_M",
                "Mean_Test_R2",
                "Median_ms_B1",
                "P95_ms_B1",
                "Median_ms_B64",
                "Throughput_images_s_B64",
            ]
        ]
        .round(4)
        .to_string()
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")

    for name, row in summary.iterrows():
        label = name.replace("vjepa_", "").capitalize()

        for ax, metric in zip(axes, ["Median_ms_B1", "Throughput_images_s_B64"]):
            if pd.notna(row[metric]):
                ax.scatter(row[metric], row["Mean_Test_R2"], s=75)
                ax.annotate(
                    label,
                    (row[metric], row["Mean_Test_R2"]),
                    xytext=(6, 5),
                    textcoords="offset points",
                )

    axes[0].set(
        xlabel="Batch-1 latency (ms)",
        ylabel="Mean clipped test R²",
        title="Prediction quality vs latency",
    )
    axes[1].set(
        xlabel="Batch-64 throughput (images/s)",
        ylabel="Mean clipped test R²",
        title="Prediction quality vs throughput",
    )

    for ax in axes:
        ax.grid(alpha=0.2)

    fig.suptitle("V-JEPA family · Prediction quality vs inference cost")

    bench_dir.mkdir(parents=True, exist_ok=True)

    with csv_path.open("x") as file:
        results.to_csv(file, index=False)

    with plot_path.open("xb") as file:
        fig.savefig(file, format="png", dpi=180)

    plt.show()
    plt.close(fig)

    print(f"\nSaved: {csv_path}")
    print(f"Saved: {plot_path}")
    print("Scope: BF16 encoder forward + mean-token pooling.")
    print("Preprocessing and 16-frame expansion excluded from timing.")
    print("Predictor skipped; total parameters include unused predictor.")
    print("Baselines: frozen pretrained encoders.")
    print("Distilled student: additional unlabeled distillation training.")

    return results, summary
