import gc
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from transformers import AutoImageProcessor, AutoModel


def benchmark_convnext(config, splits):

    models = {
        "convnext_nano": "facebook/convnextv2-nano-22k-224",
        "convnext_tiny": "facebook/convnextv2-tiny-22k-224",
        "convnext_base": "facebook/convnextv2-base-22k-224",
        "convnext_large": "facebook/convnextv2-large-22k-224",
    }

    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    device = "cuda"
    bench_dir = Path(config["output_dir"]) / "benchmarks"
    output = bench_dir / "convnext_family_inference.csv"

    if output.exists():
        raise FileExistsError(f"Already exists: {output}")

    images = np.load(
        Path(config["output_dir"]) / "data" / "hsc_images.npy",
        mmap_mode="r",
    )
    image = np.asarray(images[splits["test"][0]])

    def measure(model, pixel, batch):
        x = pixel.expand(batch, -1, -1, -1).contiguous()

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for _ in range(10):
                model(pixel_values=x).last_hidden_state.mean((2, 3))
            torch.cuda.synchronize()

            times = []
            for _ in range(40):
                torch.cuda.synchronize()
                start = time.perf_counter()
                model(pixel_values=x).last_hidden_state.mean((2, 3))
                torch.cuda.synchronize()
                times.append((time.perf_counter() - start) * 1000)

        median = float(np.median(times))
        del x
        return median, float(np.percentile(times, 95)), batch * 1000 / median

    rows = []
    print("=== CONVNEXT FAMILY · INFERENCE BENCHMARK ===")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("BF16 | 10 warmups | 40 timed runs | GPU forward only\n")

    for name, checkpoint in models.items():
        print(f"Benchmarking {name} ...")

        processor = AutoImageProcessor.from_pretrained(checkpoint)
        model = AutoModel.from_pretrained(checkpoint).to(device).eval()
        pixel = processor(images=image, return_tensors="pt")["pixel_values"].to(device)

        params = sum(p.numel() for p in model.parameters()) / 1e6

        for batch in (1, 64):
            median, p95, throughput = measure(model, pixel, batch)
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
                }
            )
            print(f"  Batch {batch:>2}: {median:.3f} ms | {throughput:,.1f} images/s")

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
    summary = summary.join(quality).reindex(models)

    assert summary.notna().all().all()

    bench_dir.mkdir(parents=True, exist_ok=True)
    with output.open("x") as file:
        results.to_csv(file, index=False)

    print("\n=== FINAL CONVNEXT FAMILY COMPARISON ===")
    print(
        summary[
            [
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
        label = name.replace("convnext_", "").capitalize()

        axes[0].scatter(row["Median_ms_B1"], row["Mean_Test_R2"], s=70)
        axes[0].annotate(
            label,
            (row["Median_ms_B1"], row["Mean_Test_R2"]),
            xytext=(5, 5),
            textcoords="offset points",
        )

        axes[1].scatter(row["Throughput_images_s_B64"], row["Mean_Test_R2"], s=70)
        axes[1].annotate(
            label,
            (row["Throughput_images_s_B64"], row["Mean_Test_R2"]),
            xytext=(5, 5),
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

    fig.suptitle("ConvNeXt family · Inference quality–cost trade-off")
    plt.show()
    plt.close(fig)

    print(f"\nSaved: {output}")
    print(
        "Scope: BF16 GPU forward + spatial embedding pooling; preprocessing excluded."
    )
    print("Baselines: frozen pretrained encoders.")
    print("Distilled student: additional unlabeled distillation training.")

    return results, summary
