import numpy as np


def create_data_splits(config):

    path = config["output_dir"] / "data" / "splits.npz"

    indices = np.random.default_rng(config["seed"]).permutation(config["num_galaxies"])

    expected = {
        "train": indices[:36_000],
        "val": indices[36_000:40_500],
        "test": indices[40_500:],
    }

    if path.exists():
        with np.load(path) as saved:
            splits = {name: saved[name] for name in expected}
        for name in expected:
            if not np.array_equal(splits[name], expected[name]):
                raise ValueError(f"Saved {name} split differs from expected.")
        print("Existing splits verified.")
    else:
        np.savez(path, **expected)
        splits = expected
        print("Splits created and saved.")

    assert np.array_equal(
        np.sort(np.concatenate(list(splits.values()))),
        np.arange(config["num_galaxies"]),
    )

    print("\nDATA PARTITIONS\n")
    for name, rows in splits.items():
        print(f"{name:5} | {len(rows):,} galaxies | {len(rows)/45000:.0%}")

    return splits
