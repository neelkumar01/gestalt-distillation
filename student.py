import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


def extract_student_embeddings(student, dataloaders, config):

    root = config["output_dir"]
    output = root / "student_embeddings" / "seed42"
    output.mkdir(parents=True, exist_ok=True)

    rows = []

    for stage in (12, 22):

        checkpoint = root / "training" / "seed42" / f"student_{stage}_best.pt"
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        student.load_state_dict(state["model"])
        student.eval()

        for split in ("train", "val"):
            dataset = dataloaders[split].dataset
            path = output / f"student_{stage}_{split}.npy"

            if path.exists():
                features = np.load(path, mmap_mode="r")
                assert features.shape == (len(dataset), config["embedding_dim"])
                assert np.isfinite(features).all()
                print(f"Verified existing: {path.name}")
                continue

            loader = DataLoader(
                dataset,
                batch_size=64,
                shuffle=False,
                num_workers=0,
                pin_memory=True,
            )

            temp = output / f"{path.stem}.partial.npy"
            if temp.exists():
                raise FileExistsError(temp)

            features = np.lib.format.open_memmap(
                temp,
                mode="w+",
                dtype=np.float32,
                shape=(len(dataset), config["embedding_dim"]),
            )

            start = time.perf_counter()
            offset = 0

            with torch.inference_mode():
                for images, _ in tqdm(loader, desc=f"Student-{stage} · {split}"):
                    images = images.to(config["device"], non_blocking=True)

                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        embeddings = student(images)

                    batch = embeddings.float().cpu().numpy()
                    assert np.isfinite(batch).all()

                    features[offset : offset + len(batch)] = batch
                    offset += len(batch)

            assert offset == len(dataset)
            elapsed = time.perf_counter() - start

            features.flush()
            del features
            temp.rename(path)

            rows.append(
                {
                    "Student": f"Student-{stage}",
                    "Split": split,
                    "Galaxies": len(dataset),
                    "Dimension": config["embedding_dim"],
                    "Seconds": round(elapsed, 1),
                }
            )

    report = pd.DataFrame(rows)
    if not report.empty:
        print("\nEMBEDDING EXTRACTION\n")
        print(report.to_string(index=False))

    print("\nSaved in:", output)
    print("Test split: NOT ACCESSED")

    return output
