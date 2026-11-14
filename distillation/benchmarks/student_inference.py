import numpy as np
import pandas as pd
import torch

def benchmark_student_inference(student, dataloaders, config):

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for this benchmark.")

    out = config["output_dir"] / "benchmarks" / "seed42_student22_inference.csv"
    if out.exists():
        raise FileExistsError(out)

    checkpoint = config["output_dir"] / "training/seed42/student_22_best.pt"
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    student.load_state_dict(state["model"])
    student.to("cuda").eval()

    batch_sizes = (1, 16, 64, 128)
    warmup, repetitions = 10, 40

    dataset = dataloaders["test"].dataset
    images = torch.stack([dataset[i][0] for i in range(max(batch_sizes))]).to("cuda")

    parameters = sum(p.numel() for p in student.parameters())
    results = []

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for batch_size in batch_sizes:
            batch = images[:batch_size]

            for _ in range(warmup):
                _ = student(batch)

            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            baseline = torch.cuda.memory_allocated()

            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            timings = []

            for _ in range(repetitions):
                start.record()
                output = student(batch)
                end.record()
                end.synchronize()

                assert output.shape == (batch_size, config["embedding_dim"])
                timings.append(start.elapsed_time(end))
                del output

            timings = np.asarray(timings)

            results.append(
                {
                    "Batch": batch_size,
                    "Median_ms": float(np.median(timings)),
                    "P95_ms": float(np.percentile(timings, 95)),
                    "Throughput_images_s": float(batch_size * 1000 / timings.mean()),
                    "Peak_GPU_GiB": float(torch.cuda.max_memory_allocated() / 1024**3),
                    "Forward_peak_GiB": float(
                        (torch.cuda.max_memory_allocated() - baseline) / 1024**3
                    ),
                    "Parameters": parameters,
                    "Warmup": warmup,
                    "Repetitions": repetitions,
                }
            )

    report = pd.DataFrame(results)

    assert np.isfinite(report.select_dtypes(include="number").to_numpy()).all()

    with out.open("x") as f:
        report.to_csv(f, index=False)

    print("\nSTUDENT-22 · GPU INFERENCE\n")
    print("GPU:", torch.cuda.get_device_name())
    print("Precision: BF16 | Input: 224×224")
    print(f"Parameters: {parameters / 1e6:.2f}M")
    print(f"Warmup: {warmup} | Timed runs: {repetitions}\n")

    print(
        report.drop(columns=["Parameters", "Warmup", "Repetitions"])
        .round(3)
        .to_string(index=False)
    )

    print("\nTiming: CUDA events, model forward only")
    print("Input preprocessing / CPU → GPU transfer: excluded")
    print("\nSaved:", out)

    return report
