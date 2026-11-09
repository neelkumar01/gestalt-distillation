import json
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download
from tqdm.auto import tqdm

def prepare_teacher_embeddings(config):

    out = config["output_dir"] / "teacher_embeddings"
    out.mkdir(parents=True, exist_ok=True)

    basket = {
        "astropt": ["015M", "095M", "850M"],
        "ijepa": ["huge", "giant"],
        "vjepa": ["large", "huge", "giant"],
        "vit": ["base", "large", "huge"],
        "vit-mae": ["base", "large", "huge"],
        "clip": ["base", "large"],
        "llava_15": ["7b", "13b"],
        "convnext": ["nano", "tiny", "base", "large"],
    }
    models = [f"{f}_{s}" for f, sizes in basket.items() for s in sizes]
    assert len(models) == 22

    manifest_path = out / "source.json"
    
    if manifest_path.exists():
        source_info = json.loads(manifest_path.read_text())
        assert source_info["repo"] == config["teacher_repo"]
        assert source_info["models"] == models
        assert source_info["rows"] == config["num_galaxies"]
    else:
        source_info = {
            "repo": config["teacher_repo"],
            "revision": HfApi().dataset_info(config["teacher_repo"]).sha,
            "models": models,
            "rows": config["num_galaxies"],
        }
        with open(manifest_path, "x") as f:
            json.dump(source_info, f, indent=2)

    records = []
    n = config["num_galaxies"]
    tag = "cosmosweb-hsc-jwst-high-snr-pil2"

    for name in tqdm(models, desc="Teachers"):
        path = out / f"{name}.npy"
        temp = out / f"{name}.partial.npy"

        if not path.exists():
            if temp.exists():
                raise FileExistsError(f"Incomplete file requires attention: {temp}")

            filename = f"cosmosweb/hsc_embeddings_{tag}_{name}_{n}.parquet"
            source = hf_hub_download(
                repo_id=config["teacher_repo"],
                repo_type="dataset",
                filename=filename,
                revision=source_info["revision"],
            )

            parquet = pq.ParquetFile(source)
            assert parquet.metadata.num_rows == n

            offset, array = 0, None
            try:
                for batch in parquet.iter_batches(
                    batch_size=512, columns=["embeddings"]
                ):
                    block = np.asarray(batch.column(0).to_pylist(), dtype=np.float32)
                    assert block.ndim == 2 and np.isfinite(block).all()

                    if array is None:
                        array = np.lib.format.open_memmap(
                            temp,
                            mode="w+",
                            dtype=np.float32,
                            shape=(n, block.shape[1]),
                        )

                    assert block.shape[1] == array.shape[1]
                    array[offset : offset + len(block)] = block
                    offset += len(block)

                assert offset == n
                array.flush()
                del array
                temp.rename(path)
            except BaseException:
                del array
                temp.unlink(missing_ok=True)
                raise

        X = np.load(path, mmap_mode="r")
        assert X.ndim == 2 and X.shape[0] == n
        for start in range(0, n, 2048):
            assert np.isfinite(X[start : start + 2048]).all()

        records.append(
            {
                "Teacher": name,
                "Dimensions": X.shape[1],
                "Rows": X.shape[0],
            }
        )

    report = pd.DataFrame(records)
    print(f"\nVerified: {len(report)}/22 teachers")
    print("Revision:", source_info["revision"][:12])
    print(report.to_string(index=False))

    return out, report
