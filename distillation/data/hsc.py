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

from datasets import load_dataset

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
