"""
GALAXY PROPERTIES
Redshift	        lephare_photozs
Stellar mass	    lp_mass
sSFR	            lp_ssfr
SFR	                lp_sfr
Stellar age	        lp_age
HSC g−r color	    mag_model_hsc-g − mag_model_hsc-r
Bulge radius	    bulge_radius
Disk radius	        disk_radius

INPUT IMAGE
hsc_images          Galaxy images used to train the DINOv3 student

ADDITIONAL COLUMNS
ra, dec             Galaxy sky coordinates.
photoz, specz       Photometric and spectroscopic redshift information
"""

import json
import numpy as np
import pandas as pd
from datasets import load_dataset
from tqdm.auto import tqdm

HSC_METADATA_COLUMNS = [
    "ra",
    "dec",
    "photoz",
    "specz",
    "lp_mass",
    "lp_ssfr",
    "lephare_photozs",
    "lp_sfr",
    "lp_age",
    "mag_model_hsc-g",
    "mag_model_hsc-r",
    "bulge_radius",
    "disk_radius",
]


def load_hsc_dataset(config):
    columns = ["hsc_images", *HSC_METADATA_COLUMNS]

    dataset = load_dataset(
        config["dataset"],
        revision=config["dataset_revision"],
        split="train",
        streaming=True,
    ).select_columns(columns)

    sample = next(iter(dataset))
    image = sample["hsc_images"].convert("RGB")

    assert image.size == (39, 39), f"Unexpected size: {image.size}"

    print("\nHSC DATASET\n")
    print("Revision:", config["dataset_revision"][:12])
    print("Image:", image.size, image.mode)
    print("Columns:", ", ".join(columns))
    print("Total columns:", len(columns))

    return dataset


def prepare_galaxy_data(config):
    out = config["output_dir"] / "data"
    out.mkdir(parents=True, exist_ok=True)

    paths = {
        "images": out / "hsc_images.npy",
        "metadata": out / "metadata.npy",
        "properties": out / "galaxy_properties.npz",
        "manifest": out / "manifest.json",
    }

    if any(p.exists() for p in paths.values()):
        raise FileExistsError("Galaxy data already exists; refusing overwrite.")

    columns = HSC_METADATA_COLUMNS
    dataset = load_hsc_dataset(config)

    n = config["num_galaxies"]
    temporary = out / "hsc_images.partial.npy"

    if temporary.exists():
        raise FileExistsError(temporary)

    images = np.lib.format.open_memmap(
        temporary,
        mode="w+",
        dtype=np.uint8,
        shape=(n, 39, 39, 3),
    )

    metadata = np.full(
        (n, len(columns)),
        np.nan,
        dtype=np.float64,
    )

    try:
        count = 0

        for i, row in enumerate(tqdm(dataset.take(n), total=n, desc="Galaxies")):
            image = np.asarray(
                row["hsc_images"].convert("RGB"),
                dtype=np.uint8,
            )

            if image.shape != (39, 39, 3):
                raise ValueError(f"Invalid image at row {i}: {image.shape}")

            images[i] = image

            metadata[i] = [
                float(row[c]) if row[c] is not None else np.nan for c in columns
            ]

            count += 1

        if count != n:
            raise ValueError(f"Expected {n:,} galaxies, received {count:,}")

        images.flush()

    except BaseException:
        del images
        temporary.unlink(missing_ok=True)
        raise

    del images

    def col(name):
        return metadata[:, columns.index(name)]

    properties = {
        "Redshift": col("lephare_photozs"),
        "Stellar mass": col("lp_mass"),
        "sSFR": col("lp_ssfr"),
        "SFR": col("lp_sfr"),
        "Stellar age": col("lp_age"),
        "HSC g-r color": (col("mag_model_hsc-g") - col("mag_model_hsc-r")),
        "Bulge radius": col("bulge_radius"),
        "Disk radius": col("disk_radius"),
    }

    np.save(paths["metadata"], metadata)
    np.savez(paths["properties"], **properties)
    temporary.rename(paths["images"])

    manifest = {
        "dataset": config["dataset"],
        "revision": config["dataset_revision"],
        "rows": n,
        "split": "train",
        "row_order": "Original first 45000 dataset rows",
        "metadata_columns": columns,
        "image_shape": [39, 39, 3],
    }

    with open(paths["manifest"], "x") as f:
        json.dump(manifest, f, indent=2)

    summary = pd.DataFrame(
        {
            "Property": list(properties),
            "Valid galaxies": [int(np.isfinite(y).sum()) for y in properties.values()],
        }
    )

    print(f"Prepared {n:,} aligned galaxies")
    print(f"Images: ({n}, 39, 39, 3)")
    print("\nTarget availability:")
    print(summary.to_string(index=False))
    print("\nSaved to:", out)

    return out, properties
