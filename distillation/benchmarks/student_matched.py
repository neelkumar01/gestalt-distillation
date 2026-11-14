import time
import numpy as np
import pandas as pd
import torch


def benchmark_student22_matched(student, dataloaders, config):

    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    output = (
        config["output_dir"] / "benchmarks" / "seed42_student22_inference_matched.csv"
    )
    if output.exists():
        raise FileExistsError(output)

    checkpoint = config["output_dir"] / "training/seed42/student_22_best.pt"
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    student.load_state_dict(state["model"])
    student.to("cuda").eval()

    image = dataloaders["test"].dataset[0][0].unsqueeze(0).to("cuda")
    parameters = sum(p.numel() for p in student.parameters()) / 1e6
    rows = []

    print("=== STUDENT-22 · MATCHED INFERENCE BENCHMARK ===")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("BF16 | 10 warmups | 40 timed runs | Batch 1 & 64")
    print("224×224 | GPU forward only\n")

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for batch in (1, 64):
            x = image.expand(batch, -1, -1, -1).contiguous()

            assert student(x).shape == (batch, config["embedding_dim"])

            for _ in range(10):
                student(x)

            torch.cuda.synchronize()
            timings = []

            for _ in range(40):
                torch.cuda.synchronize()
                start = time.perf_counter()

                student(x)

                torch.cuda.synchronize()
                timings.append((time.perf_counter() - start) * 1000)

            median = float(np.median(timings))
            p95 = float(np.percentile(timings, 95))
            throughput = batch * 1000 / median

            rows.append(
                {
                    "Model": "Student-22",
                    "Batch": batch,
                    "Median_ms": median,
                    "P95_ms": p95,
                    "Throughput_images_s": throughput,
                    "Parameters_M": parameters,
                    "Precision": "BF16",
                    "GPU": torch.cuda.get_device_name(0),
                }
            )

            print(
                f"Batch {batch:>2}: {median:.3f} ms | "
                f"P95 {p95:.3f} ms | "
                f"{throughput:,.1f} images/s"
            )
            del x

    results = pd.DataFrame(rows)
    assert (
        np.isfinite(results[["Median_ms", "P95_ms", "Throughput_images_s"]]).all().all()
    )

    with output.open("x") as file:
        results.to_csv(file, index=False)

    print(f"\nParameters: {parameters:.2f}M")
    print("Checkpoint: student_22_best.pt")
    print("Input preprocessing and transfer excluded.")
    print("Saved:", output)

    return results
