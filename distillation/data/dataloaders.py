import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


def create_dataloaders(config, splits, teacher_target_dir):

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )

    images = np.load(
        config["output_dir"] / "data/hsc_images.npy",
        mmap_mode="r",
    )
    targets = np.load(
        teacher_target_dir / "gestalt_1024.npy",
        mmap_mode="r",
    )

    class GalaxyDataset(Dataset):
        def __init__(self, indices):
            self.indices = indices

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, i):
            row = int(self.indices[i])
            image = transform(Image.fromarray(np.asarray(images[row])))
            target = torch.from_numpy(np.array(targets[row], copy=True))
            return image, target

    loaders = {
        name: DataLoader(
            GalaxyDataset(indices),
            batch_size=64,
            shuffle=(name == "train"),
            generator=torch.Generator().manual_seed(config["seed"]),
            num_workers=0,
            pin_memory=True,
        )
        for name, indices in splits.items()
    }

    sample_images, sample_targets = next(iter(loaders["val"]))

    assert sample_images.shape == (64, 3, 224, 224)
    assert sample_targets.shape == (64, 1024)
    assert torch.isfinite(sample_images).all()
    assert torch.isfinite(sample_targets).all()

    print("\nSTUDENT DATA PIPELINE\n")
    print("Input: 224×224 RGB | ImageNet normalization")
    print("Teacher target: 1024D | Batch size: 64")
    print()
    print(f"{'Split':<12}{'Galaxies':>10}{'Batches':>10}")
    for name, loader in loaders.items():
        print(f"{name:<12}{len(loader.dataset):>10,}{len(loader):>10,}")
    print("\nSample batch:", tuple(sample_images.shape))
    print("Target batch:", tuple(sample_targets.shape))

    return loaders
