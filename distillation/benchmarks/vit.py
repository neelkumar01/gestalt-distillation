import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from transformers import AutoImageProcessor, AutoModel


def benchmark_vit(config, splits):

    models = {
        "vit_base": "google/vit-base-patch16-224-in21k",
        "vit_large": "google/vit-large-patch16-224-in21k",
        "vit_huge": "google/vit-huge-patch14-224-in21k",
    }

    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    bench_dir = Path(config["output_dir"]) / "benchmarks"
    csv_path = bench_dir / "vit_family_inference.csv"
    fig_path = bench_dir / "vit_family_quality_cost.png"

    if csv_path.exists() or fig_path.exists():
        raise FileExistsError("ViT family benchmark artifacts already exist.")

    images = np.load(
        Path(config["output_dir"]) / "data" / "hsc_images.npy",
        mmap_mode="r",
    )
    image = np.asarray(images[splits["test"][0]])

    def measure(model, pixel, batch):
        x = pixel.expand(batch, -1, -1, -1).contiguous()

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for _ in range(10):
                model(pixel_values=x).last_hidden_state[:, 1:].mean(dim=1)
            torch.cuda.synchronize()

            times = []
            for _ in range(40):
                torch.cuda.synchronize()
                start = time.perf_counter()
                model(pixel_values=x).last_hidden_state[:, 1:].mean(dim=1)
                torch.cuda.synchronize()
                times.append((time.perf_counter() - start) * 1000)

        median = float(np.median(times))
        return median, float(np.percentile(times, 95)), batch * 1000 / median

    rows = []

    print("=== ViT FAMILY · INFERENCE BENCHMARK ===")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("BF16 | 10 warmups | 40 timed runs | GPU forward only\n")

    for name, checkpoint in models.items():
        print(f"Benchmarking {name} ...")

        processor = AutoImageProcessor.from_pretrained(checkpoint)
        model = AutoModel.from_pretrained(checkpoint).to("cuda").eval()
        pixel = processor(images=image, return_tensors="pt")["pixel_values"].to("cuda")

        params = sum(p.numel() for p in model.parameters()) / 1e6

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            embedding = model(pixel_values=pixel).last_hidden_state[:, 1:].mean(dim=1)

        cached = np.load(
            Path(config["output_dir"]) / "teacher_embeddings" / f"{name}.npy",
            mmap_mode="r",
        )
        assert cached.shape == (config["num_galaxies"], embedding.shape[1])
        del embedding, cached

        for batch in (1, 64):
            try:
                median, p95, throughput = measure(model, pixel, batch)
                status = "OK"
            except torch.cuda.OutOfMemoryError:
                if batch == 1:
                    raise
                median = p95 = throughput = float("nan")
                status = "OOM"
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
                    "Precision": "BF16",
                    "GPU": torch.cuda.get_device_name(0),
                    "Status": status,
                }
            )

            if status == "OK":
                print(
                    f"  Batch {batch:>2}: {median:.3f} ms | {throughput:,.1f} images/s"
                )
            else:
                print(f"  Batch {batch:>2}: OOM — not measured")

        del model, processor, pixel
        gc.collect()
        torch.cuda.empty_cache()

    results = pd.DataFrame(rows)

    quality = pd.read_csv(bench_dir / "seed42_exp1_tuned_test_r2.csv")
    quality = (
        quality[(quality["Protocol"] == "Clipped") & quality["Model"].isin(models)]
        .groupby("Model")["Test_R2"]
        .mean()
        .rename("Mean_Test_R2")
    )

    summary = results.pivot(
        index="Model",
        columns="Batch",
        values=["Median_ms", "Throughput_images_s"],
    )
    summary.columns = [f"{metric}_B{batch}" for metric, batch in summary.columns]
    summary = (
        summary.join(quality)
        .join(results.groupby("Model")["Parameters_M"].first())
        .reindex(models)
    )

    assert summary["Mean_Test_R2"].notna().all()
    assert summary["Median_ms_B1"].notna().all()

    bench_dir.mkdir(parents=True, exist_ok=True)
    with csv_path.open("x") as file:
        results.to_csv(file, index=False)

    print("\n=== FINAL ViT FAMILY COMPARISON ===")
    print(
        summary[
            [
                "Parameters_M",
                "Mean_Test_R2",
                "Median_ms_B1",
                "Median_ms_B64",
                "Throughput_images_s_B64",
            ]
        ]
        .round(4)
        .to_string()
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")

    for name, row in summary.iterrows():
        label = name.replace("vit_", "").capitalize()

        for ax, metric in zip(
            axes,
            ["Median_ms_B1", "Throughput_images_s_B64"],
        ):
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

    fig.suptitle("ViT family · Inference quality–cost trade-off")
    with fig_path.open("xb") as file:
        fig.savefig(file, format="png", dpi=180)

    plt.show()
    plt.close(fig)

    print(f"\nSaved: {csv_path}")
    print(f"Saved: {fig_path}")
    print("Scope: BF16 GPU forward + patch-token mean pooling; preprocessing excluded.")
    print("Baselines: frozen pretrained encoders.")
    print("Distilled student: additional unlabeled distillation training.")

    return results, summary
