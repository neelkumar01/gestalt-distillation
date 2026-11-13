import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


def prepare_test_embeddings(
    student, dataloaders, config, splits, student_embeddings_dir
):

    root = config["output_dir"]
    benchmark_dir = root / "benchmarks"
    selection_path = benchmark_dir / "validation_selected_teachers.json"

    scores = pd.read_csv(benchmark_dir / "individual_teachers_clipped_val.csv")
    winners = (
        scores.sort_values(
            ["Property", "R2", "Teacher"],
            ascending=[True, False, True],
        )
        .drop_duplicates("Property")
        .set_index("Property")["Teacher"]
        .to_dict()
    )
    assert len(winners) == 8

    if selection_path.exists():
        assert json.loads(selection_path.read_text()) == winners
    else:
        with selection_path.open("x") as f:
            json.dump(winners, f, indent=2)

    n_test = len(splits["test"])
    dim = config["embedding_dim"]

    loader = DataLoader(
        dataloaders["test"].dataset,
        batch_size=64,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    for stage in (12, 22):
        path = student_embeddings_dir / f"student_{stage}_test.npy"

        if path.exists():
            saved = np.load(path, mmap_mode="r")
            assert saved.shape == (n_test, dim)
            assert np.isfinite(saved).all()
            print(f"Student-{stage}: existing test embeddings verified")
            continue

        checkpoint = root / "training" / "seed42" / f"student_{stage}_best.pt"
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        student.load_state_dict(state["model"])
        student.eval()

        temp = path.with_suffix(".partial.npy")
        if temp.exists():
            raise FileExistsError(temp)

        features = np.lib.format.open_memmap(
            temp, mode="w+", dtype=np.float32, shape=(n_test, dim)
        )

        offset = 0
        with torch.inference_mode():
            for images, _ in tqdm(loader, desc=f"Student-{stage} · test"):
                images = images.to(config["device"], non_blocking=True)

                with torch.autocast("cuda", dtype=torch.bfloat16):
                    embeddings = student(images)

                batch = embeddings.float().cpu().numpy()
                assert np.isfinite(batch).all()

                features[offset : offset + len(batch)] = batch
                offset += len(batch)

        assert offset == n_test
        features.flush()
        del features
        temp.rename(path)

        print(f"Student-{stage}: saved {n_test:,} × {dim:,}")

    return selection_path
