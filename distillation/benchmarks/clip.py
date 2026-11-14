import gc
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from transformers import CLIPModel, CLIPProcessor


def benchmark_clip(config, splits):

    models = {
        "clip_base": "openai/clip-vit-base-patch16",
        "clip_large": "openai/clip-vit-large-patch14",
    }
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    bench_dir = Path(config["output_dir"]) / "benchmarks"
    csv_path = bench_dir / "clip_family_inference.csv"
    plot_path = bench_dir / "clip_family_quality_cost.png"

    if csv_path.exists() or plot_path.exists():
        raise FileExistsError("CLIP family outputs already exist.")

    quality = pd.read_csv(bench_dir / "seed42_exp1_tuned_test_r2.csv")
    quality = quality[
        quality["Protocol"].str.lower().eq("clipped") & quality["Model"].isin(models)
    ]
    scores = quality.groupby("Model")["Test_R2"].agg(["mean", "count"])
    assert (
        set(scores.index) == set(models) and scores["count"].eq(8).all()
    ), "Expected eight clipped test properties per CLIP model."

    images = np.load(
        Path(config["output_dir"]) / "data" / "hsc_images.npy",
        mmap_mode="r",
    )
    image = np.asarray(images[splits["test"][0]])

    def features(model, x):
        out = model.get_image_features(pixel_values=x)
        if isinstance(out, torch.Tensor):
            return out
        return out.pooler_output

    def measure(model, pixel, batch):
        x = pixel.expand(batch, -1, -1, -1).contiguous()

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for _ in range(10):
                features(model, x)

            torch.cuda.synchronize()
            times = []

            for _ in range(40):
                torch.cuda.synchronize()
                start = time.perf_counter()

                features(model, x)

                torch.cuda.synchronize()
                times.append((time.perf_counter() - start) * 1000)

        median = float(np.median(times))
        p95 = float(np.percentile(times, 95))
        return median, p95, batch * 1000 / median

    gpu = torch.cuda.get_device_name(0)
    rows = []

    print("=== CLIP FAMILY · INFERENCE BENCHMARK ===")
    print(f"GPU: {gpu}")
    print("BF16 | 10 warmups | 40 timed runs | Batch 1 & 64")
    print("Projected image features | Preprocessing excluded\n")

    for name, checkpoint in models.items():
        print(f"Benchmarking {name} ...")

        processor = CLIPProcessor.from_pretrained(checkpoint)
        model = CLIPModel.from_pretrained(checkpoint).to("cuda").eval()

        pixel = processor(images=image, return_tensors="pt")["pixel_values"].to("cuda")

        total_params = sum(p.numel() for p in model.parameters()) / 1e6
        visual_params = (
            sum(p.numel() for p in model.vision_model.parameters())
            + sum(p.numel() for p in model.visual_projection.parameters())
        ) / 1e6

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            embedding = features(model, pixel)

        assert embedding.shape == (1, model.config.projection_dim)

        cached = np.load(
            Path(config["output_dir"]) / "teacher_embeddings" / f"{name}.npy",
            mmap_mode="r",
        )
        assert cached.shape == (
            len(images),
            embedding.shape[1],
        ), f"Cached embedding shape mismatch for {name}: {cached.shape}"
        del embedding, cached

        print(
            f"  Parameters: {total_params:.2f}M total / "
            f"{visual_params:.2f}M visual | "
            f"Patch: {model.config.vision_config.patch_size} | "
            f"Input: {pixel.shape[-2]}×{pixel.shape[-1]} | "
            f"Projection: {model.config.projection_dim}"
        )

        for batch in (1, 64):
            try:
                median, p95, throughput = measure(model, pixel, batch)
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
                    "Parameters_M": total_params,
                    "Visual_Parameters_M": visual_params,
                    "Patch_Size": model.config.vision_config.patch_size,
                    "Input_Size": pixel.shape[-1],
                    "Embedding_Dim": model.config.projection_dim,
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

        del model, processor, pixel
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
        results.groupby("Model")[["Parameters_M", "Visual_Parameters_M"]].first()
    ).reindex(models)

    summary["Mean_Test_R2"] = scores.loc[summary.index, "mean"]

    print("=== FINAL CLIP FAMILY COMPARISON ===")
    print(
        summary[
            [
                "Parameters_M",
                "Visual_Parameters_M",
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
        label = name.replace("clip_", "").capitalize()

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

    fig.suptitle("CLIP family · Prediction quality vs inference cost")

    bench_dir.mkdir(parents=True, exist_ok=True)

    with csv_path.open("x") as file:
        results.to_csv(file, index=False)

    with plot_path.open("xb") as file:
        fig.savefig(file, format="png", dpi=180)

    plt.show()
    plt.close(fig)

    print(f"\nSaved: {csv_path}")
    print(f"Saved: {plot_path}")
    print("Scope: BF16 visual encoder + projection; preprocessing excluded.")
    print("Total parameters include the unused text encoder.")
    print("Baselines frozen; distilled student received extra unlabeled training.")

    return results, summary
