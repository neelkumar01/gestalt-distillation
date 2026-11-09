import random
from pathlib import Path

import numpy as np
import torch


def initialize_experiment():
    seed = 42

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    config = {
        "dataset": "Ashodkh/cosmosweb-hsc-jwst-high-snr-pil2",
        "dataset_revision": "814515995f77fc2106bd03b55e03c3e8e9479ec8",
        "teacher_repo": "UniverseTBD/pu-embeddings",
        "student_model": "facebook/dinov3-vitb16-pretrain-lvd1689m",
        "num_galaxies": 45_000,
        "embedding_dim": 1024,
        "seed": seed,
        "device": "cuda",
        "output_dir": Path("gestalt_distillation"),
    }

    config["output_dir"].mkdir(parents=True, exist_ok=True)

    print("Dataset:", config["dataset"])
    print("Revision:", config["dataset_revision"][:12])
    print("Galaxies:", config["num_galaxies"])
    print("Student:", config["student_model"])
    print("Seed:", seed)
    print("Output:", config["output_dir"])

    return config
