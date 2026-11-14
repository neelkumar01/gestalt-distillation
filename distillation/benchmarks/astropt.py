import gc
import io
import re
import time
from contextlib import redirect_stdout
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from astropt.model_utils import load_astropt
from astropt.local_datasets import GalaxyImageDataset


def benchmark_astropt(config, splits):

    sizes = ("015M", "095M", "850M")
    repo = "Smith42/astroPT_v2.0"
    bench_dir = Path(config["output_dir"]) / "benchmarks"
    csv_path = bench_dir / "astropt_family_inference.csv"
    plot_path = bench_dir / "astropt_family_quality_cost.png"

    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    if csv_path.exists() or plot_path.exists():
        raise FileExistsError("AstroPT family outputs already exist.")

    def model_key(value):
        key = re.sub(r"[^a-z0-9]", "", str(value).lower())
        return re.sub(r"^astropt0+", "astropt", key)

    quality = pd.read_csv(bench_dir / "seed42_exp1_tuned_test_r2.csv")
    quality = quality[quality["Protocol"].str.lower().eq("clipped")].copy()
    quality["key"] = quality["Model"].map(model_key)
    scores = quality.groupby("key")["Test_R2"].mean()

    names = [f"astropt_{size}" for size in sizes]
    assert all(model_key(name) in scores for name in names)

    images = np.load(
        Path(config["output_dir"]) / "data" / "hsc_images.npy",
        mmap_mode="r",
    )
    image = (
        torch.from_numpy(np.array(images[splits["test"][0]], copy=True))
        .permute(2, 0, 1)
        .float()
    )
    image = F.interpolate(
        image[None],
        size=(96, 96),
        mode="bilinear",
        align_corners=False,
    )[0]

    def normalise(x):
        std, mean = torch.std_mean(x, dim=1, keepdim=True)
        return (x - mean) / (std + 1e-8)

    rows = []

    print("=== AstroPT FAMILY · INFERENCE BENCHMARK ===")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("BF16 | 10 warmups | 40 timed runs | Batch 1 & 64")
    print("Scope: GPU embedding inference; preprocessing excluded\n")

    for size, name in zip(sizes, names):
        print(f"Benchmarking {name} ...")

        with redirect_stdout(io.StringIO()):
            model = load_astropt(repo, path=f"astropt/{size}").to("cuda").eval()

        processor = GalaxyImageDataset(
            None,
            spiral=True,
            transform={"images": normalise},
            modality_registry=model.modality_registry,
        )

        patches = processor.process_galaxy(image)
        pixel = patches[None].to("cuda")
        positions = torch.arange(len(patches), device="cuda", dtype=torch.long)[None]

        params = sum(p.numel() for p in model.parameters()) / 1e6
        patch_size = model.modality_registry.get_config("images").patch_size

        def forward(x, pos):
            return model.generate_embeddings(
                {
                    "images": x,
                    "images_positions": pos,
                }
            )["images"]

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            with redirect_stdout(io.StringIO()):
                embedding = forward(pixel, positions)

        cache_dir = Path(config["output_dir"]) / "teacher_embeddings"
        cache = [
            p
            for p in cache_dir.glob("astropt*.npy")
            if model_key(p.stem) == model_key(name)
        ]
        assert len(cache) == 1, f"Expected one cached embedding file for {name}"
        cached_shape = np.load(cache[0], mmap_mode="r").shape
        assert cached_shape == (config["num_galaxies"], embedding.shape[-1])

        print(
            f"  Parameters: {params:.2f}M | "
            f"Patch: {patch_size} | Tokens: {len(patches)} | "
            f"Embedding: {embedding.shape[-1]}"
        )
        del embedding

        for batch in (1, 64):
            x = pixel.expand(batch, -1, -1).contiguous()
            pos = positions.expand(batch, -1).contiguous()

            try:
                with torch.inference_mode(), torch.autocast(
                    "cuda", dtype=torch.bfloat16
                ), redirect_stdout(io.StringIO()):
                    for _ in range(10):
                        forward(x, pos)
                    torch.cuda.synchronize()

                    times = []
                    for _ in range(40):
                        torch.cuda.synchronize()
                        start = time.perf_counter()
                        forward(x, pos)
                        torch.cuda.synchronize()
                        times.append((time.perf_counter() - start) * 1000)

                median = float(np.median(times))
                p95 = float(np.percentile(times, 95))
                throughput = batch * 1000 / median
                status = "OK"

            except torch.cuda.OutOfMemoryError:
                if batch == 1:
                    raise
                median = p95 = throughput = float("nan")
                status = "OOM"

            rows.append(
                {
                    "Model": name,
                    "Checkpoint": f"{repo}/astropt/{size}",
                    "Batch": batch,
                    "Median_ms": median,
                    "P95_ms": p95,
                    "Throughput_images_s": throughput,
                    "Parameters_M": params,
                    "Patch_Size": patch_size,
                    "Tokens": len(patches),
                    "Embedding_Dim": cached_shape[1],
                    "Precision": "BF16",
                    "GPU": torch.cuda.get_device_name(0),
                    "Status": status,
                }
            )

            if status == "OK":
                print(
                    f"  Batch {batch:>2}: {median:.3f} ms | "
                    f"P95 {p95:.3f} ms | {throughput:,.1f} images/s"
                )
            else:
                print(f"  Batch {batch:>2}: OOM — not measured")

            del x, pos
            torch.cuda.empty_cache()

        del model, processor, patches, pixel, positions, forward
        gc.collect()
        torch.cuda.empty_cache()
        print()

    results = pd.DataFrame(rows)

    summary = results.pivot(
        index="Model",
        columns="Batch",
        values=["Median_ms", "Throughput_images_s"],
    )
    summary.columns = [f"{metric}_B{batch}" for metric, batch in summary.columns]
    summary = summary.join(results.groupby("Model")["Parameters_M"].first()).reindex(
        names
    )
    summary["Mean_Test_R2"] = [scores.loc[model_key(name)] for name in names]

    print("=== FINAL AstroPT FAMILY COMPARISON ===")
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
        label = name.split("_")[1]

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

    fig.suptitle("AstroPT family · Prediction quality vs inference cost")

    bench_dir.mkdir(parents=True, exist_ok=True)
    with csv_path.open("x") as f:
        results.to_csv(f, index=False)
    with plot_path.open("xb") as f:
        fig.savefig(f, format="png", dpi=180)

    plt.show()
    plt.close(fig)

    print(f"\nSaved: {csv_path}")
    print(f"Saved: {plot_path}")
    print("Input: cached HSC image resized to 96×96 (timing proxy).")
    print("Scope: BF16 GPU generate_embeddings; preprocessing excluded.")
    print("Baselines: frozen pretrained encoders.")
    print("Distilled student: additional unlabeled distillation training.")

    return results, summary
