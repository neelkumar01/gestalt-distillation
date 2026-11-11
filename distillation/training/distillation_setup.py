import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def configure_distillation(student, config, splits, teacher_target_dir):

    path = teacher_target_dir / "target_normalization.npz"
    targets = np.load(teacher_target_dir / "gestalt_1024.npy", mmap_mode="r")
    train = np.asarray(targets[splits["train"]])

    mean = train.mean(axis=0).astype(np.float32)
    std = np.maximum(train.std(axis=0), 1e-6).astype(np.float32)

    if path.exists():
        with np.load(path) as saved:
            assert np.allclose(saved["mean"], mean)
            assert np.allclose(saved["std"], std)
    else:
        with open(path, "xb") as f:
            np.savez(f, mean=mean, std=std)

    class DistillationLoss(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("mean", torch.from_numpy(mean))
            self.register_buffer("std", torch.from_numpy(std))

        def forward(self, predictions, targets):
            targets = (targets.float() - self.mean) / self.std
            predictions = predictions.float()
            mse = F.mse_loss(predictions, targets)
            cosine = 1 - F.cosine_similarity(predictions, targets, dim=-1).mean()
            return mse + 0.1 * cosine

    criterion = DistillationLoss().to(config["device"])

    optimizer = torch.optim.AdamW(
        [
            {"params": student.backbone.parameters(), "lr": 1e-5},
            {"params": student.head.parameters(), "lr": 1e-4},
        ],
        weight_decay=0.05,
    )

    print("\nDISTILLATION CONFIGURATION\n")
    print("Loss: MSE + 0.1 × cosine distance")
    print("Target normalization: Train only, 1024 dimensions")
    print("Optimizer: AdamW | Weight decay: 0.05")
    print("Backbone LR: 1e-5 | Head LR: 1e-4")
    print("Normalization saved:", path)

    return criterion, optimizer
