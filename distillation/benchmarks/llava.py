import gc
import time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor


def benchmark_llava_family(config, splits):

    models = {
        "llava_15_7b": "llava-hf/llava-1.5-7b-hf",
        "llava_15_13b": "llava-hf/llava-1.5-13b-hf",
    }
    prompt = "USER: <image>\n ASSISTANT:"

    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    bench_dir = Path(config["output_dir"]) / "benchmarks"
    csv_path = bench_dir / "llava_family_inference.csv"
    plot_path = bench_dir / "llava_family_quality_cost.png"

    if csv_path.exists() or plot_path.exists():
        raise FileExistsError("LLaVA family outputs already exist.")

    quality = pd.read_csv(bench_dir / "seed42_exp1_tuned_test_r2.csv")
    quality = quality[
        quality["Protocol"].str.lower().eq("clipped") & quality["Model"].isin(models)
    ]
    scores = quality.groupby("Model")["Test_R2"].agg(["mean", "count"])
    assert (
        set(scores.index) == set(models) and scores["count"].eq(8).all()
    ), f"Expected eight clipped test properties per model; found {scores.index.tolist()}"

    images = np.load(
        Path(config["output_dir"]) / "data" / "hsc_images.npy",
        mmap_mode="r",
    )
    image = np.asarray(images[splits["test"][0]])

    def prepare_inputs(processor, model):
        first = processor(
            images=image, text="<image>", return_tensors="pt", padding=True
        )["pixel_values"][0].float()

        first = (first - first.min()) / (first.max() - first.min() + 1e-8)
        pil = Image.fromarray((first.permute(1, 2, 0) * 255).byte().numpy())
        enc = processor(images=pil, text=prompt, return_tensors="pt", padding=True)

        image_token = getattr(model.config, "image_token_index", None)
        if image_token is None:
            image_token = getattr(model.config, "image_token_id", None)

        assert image_token is not None
        n_image_tokens = int((enc["input_ids"] == image_token).sum())
        assert n_image_tokens > 0, "Prompt has no image tokens."

        return {
            "input_ids": enc["input_ids"].to("cuda"),
            "attention_mask": enc["attention_mask"].to("cuda"),
            "pixel_values": enc["pixel_values"].to("cuda", torch.bfloat16),
        }, n_image_tokens

    def forward(model, x):
        output = model(
            **x,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        hidden = output.hidden_states[-1]
        mask = x["attention_mask"].float().unsqueeze(-1)
        return (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)

    def measure(model, inputs, batch):
        x = {k: v.expand(batch, *v.shape[1:]).contiguous() for k, v in inputs.items()}

        with torch.inference_mode():
            for _ in range(10):
                forward(model, x)

            torch.cuda.synchronize()
            times = []

            for _ in range(40):
                torch.cuda.synchronize()
                start = time.perf_counter()

                forward(model, x)

                torch.cuda.synchronize()
                times.append((time.perf_counter() - start) * 1000)

        median = float(np.median(times))
        p95 = float(np.percentile(times, 95))
        return median, p95, batch * 1000 / median

    gpu = torch.cuda.get_device_name(0)
    rows = []

    print("=== LLaVA FAMILY · INFERENCE BENCHMARK ===")
    print(f"GPU: {gpu}")
    print("BF16 | 10 warmups | 40 timed runs | Batch 1 & 64")
    print("Full multimodal forward | Final LLM masked-mean embedding")
    print("KV cache disabled | Preprocessing excluded\n")

    for name, checkpoint in models.items():
        print(f"Benchmarking {name} ...")

        gc.collect()
        torch.cuda.empty_cache()

        processor = AutoProcessor.from_pretrained(checkpoint)
        model = AutoModelForImageTextToText.from_pretrained(
            checkpoint,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True,
        ).eval()

        assert all(
            p.device.type == "cuda" for p in model.parameters()
        ), f"{name}: CPU/disk offload detected; GPU-only comparison invalid."

        inputs, image_tokens = prepare_inputs(processor, model)
        params = sum(p.numel() for p in model.parameters()) / 1e6

        with torch.inference_mode():
            embedding = forward(model, inputs)

        assert embedding.ndim == 2 and embedding.shape[0] == 1
        assert torch.isfinite(embedding).all(), "Non-finite embedding."

        print(
            f"  Parameters: {params:,.2f}M | "
            f"Embedding: {embedding.shape[1]} | "
            f"Image tokens: {image_tokens}"
        )
        print(
            f"  Input: {inputs['pixel_values'].shape[-2]}×"
            f"{inputs['pixel_values'].shape[-1]} | "
            f"Sequence: {inputs['input_ids'].shape[1]} tokens"
        )
        del embedding

        for batch in (1, 64):
            try:
                median, p95, throughput = measure(model, inputs, batch)
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
                    "Embedding_Dim": model.config.text_config.hidden_size,
                    "Image_Tokens": image_tokens,
                    "Sequence_Length": inputs["input_ids"].shape[1],
                    "Input_Size": inputs["pixel_values"].shape[-1],
                    "Precision": "BF16",
                    "KV_Cache": False,
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

        del model, processor, inputs
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

    summary = summary.join(results.groupby("Model")["Parameters_M"].first()).reindex(
        models
    )
    summary["Mean_Test_R2"] = scores.loc[summary.index, "mean"]

    print("=== FINAL LLaVA FAMILY COMPARISON ===")
    print(
        summary[
            [
                "Parameters_M",
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
        label = name.replace("llava_15_", "").upper()
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

    fig.suptitle("LLaVA family · Prediction quality vs inference cost")

    bench_dir.mkdir(parents=True, exist_ok=True)

    with csv_path.open("x") as file:
        results.to_csv(file, index=False)

    with plot_path.open("xb") as file:
        fig.savefig(file, format="png", dpi=180)

    plt.show()
    plt.close(fig)

    print(f"\nSaved: {csv_path}")
    print(f"Saved: {plot_path}")
    print("Scope: full multimodal embedding forward + masked pooling.")
    print("Preprocessing/model loading excluded; no generation.")
    print("KV cache disabled; CPU offloading disallowed.")
    print("Student baselines received extra unlabeled distillation training.")

    return results, summary
