import json
import numpy as np
import torch
from tqdm.auto import tqdm


def build_gestalt_targets(config, splits, whitened_dir, model_names):

    out = config["output_dir"] / "teacher"
    out.mkdir(parents=True, exist_ok=True)

    target_path = out / "gestalt_1024.npy"
    projector_path = out / "mcca_projector.npy"
    order_path = out / "model_order.json"

    if any(p.exists() for p in (target_path, projector_path, order_path)):
        raise FileExistsError("Existing Gestalt outputs; refusing overwrite.")

    views = [
        np.load(whitened_dir / f"{name}.npy", mmap_mode="r") for name in model_names
    ]
    widths = [v.shape[1] for v in views]
    train = splits["train"]
    d = config["embedding_dim"]

    X = torch.empty((len(train), sum(widths)), device="cuda", dtype=torch.float32)

    offset = 0
    for view, width in tqdm(
        zip(views, widths), total=len(views), desc="Combining teachers"
    ):
        X[:, offset : offset + width] = torch.from_numpy(np.asarray(view[train])).to(
            "cuda"
        )
        offset += width

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    with torch.no_grad():
        _, singular_values, V = torch.svd_lowrank(X, q=d + 64, niter=4)
        V = V[:, :d].contiguous()
        explained = (singular_values[:d].square().sum() / X.square().sum()).item()
        orth_error = (
            torch.linalg.matrix_norm(V.T @ V - torch.eye(d, device="cuda")).item() / d
        )

        assert orth_error < 1e-4
        del X

        temp = out / "gestalt_1024.partial.npy"
        if temp.exists():
            raise FileExistsError(temp)

        targets = np.lib.format.open_memmap(
            temp,
            mode="w+",
            dtype=np.float32,
            shape=(config["num_galaxies"], d),
        )

        for start in tqdm(range(0, len(targets), 512), desc="Projecting galaxies"):
            end = min(start + 512, len(targets))
            batch = torch.cat(
                [torch.from_numpy(np.asarray(v[start:end])).to("cuda") for v in views],
                dim=1,
            )

            z = (batch @ V).cpu().numpy()
            assert np.isfinite(z).all()
            targets[start:end] = z

        targets.flush()
        del targets

        np.save(projector_path, V.cpu().numpy())
        temp.rename(target_path)
        order_path.write_text(
            json.dumps(
                {
                    "models": model_names,
                    "widths": widths,
                    "fit_split": "train",
                    "svd_seed": 0,
                    "svd_q": d + 64,
                    "svd_iterations": 4,
                },
                indent=2,
            )
        )

    print("\nGESTALT-22 TEACHER\n")
    print(f"Input: 22 teachers × {sum(widths):,} total dimensions")
    print(f"Output: {config['num_galaxies']:,} × {d:,}")
    print(f"Train only explained energy: {explained:.2%}")
    print(f"Projector orthogonality error: {orth_error:.2e}")
    print("Fit data: train only")
    print("Saved:", out)

    return out
