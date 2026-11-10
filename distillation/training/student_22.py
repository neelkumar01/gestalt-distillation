import math
import pandas as pd
import matplotlib.pyplot as plt
import torch
from tqdm.auto import tqdm


def train_student_22(student, dataloaders, criterion, config):

    out = config["output_dir"] / "training" / "seed42"
    base = out / "student_12_best.pt"
    best = out / "student_22_best.pt"
    last = out / "student_22_last.pt"
    if not base.exists():
        raise FileNotFoundError(base)

    optimizer = torch.optim.AdamW(
        [
            {"params": student.backbone.parameters(), "lr": 1e-5},
            {"params": student.head.parameters(), "lr": 1e-4},
        ],
        weight_decay=0.05,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    def save(state, path):
        temp = path.with_suffix(".tmp")
        torch.save(state, temp)
        temp.replace(path)

    if last.exists():
        state = torch.load(last, map_location="cpu", weights_only=True)
        student.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        dataloaders["train"].generator.set_state(state["loader_rng"])
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])
        history = state["history"]
        best_loss = state["best_loss"]
        start = state["epoch"] + 1
        print(f"Resuming at epoch {start}")
    else:
        if best.exists():
            raise FileExistsError(best)
        state = torch.load(base, map_location="cpu", weights_only=True)
        student.load_state_dict(state["model"])
        best_loss = float("inf")
        history, start = [], 13
        print("Continuing from best Student-12 checkpoint")

    for epoch in range(start, 23):
        metrics = {}

        for phase in ("train", "val"):
            training = phase == "train"
            student.train(training)
            total = count = 0

            for images, targets in tqdm(
                dataloaders[phase],
                desc=f"Epoch {epoch}/22 · {phase}",
                leave=False,
            ):
                images = images.to(config["device"], non_blocking=True)
                targets = targets.to(config["device"], non_blocking=True)

                if training:
                    optimizer.zero_grad(set_to_none=True)

                with torch.set_grad_enabled(training):
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        predictions = student(images)

                    loss = criterion(predictions.float(), targets)
                    if not math.isfinite(loss.item()):
                        raise FloatingPointError(f"Non-finite loss: epoch {epoch}")

                    if training:
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                        optimizer.step()

                total += loss.item() * len(images)
                count += len(images)

            metrics[phase] = total / count

        history.append({"epoch": epoch, **metrics})

        if metrics["val"] < best_loss:
            best_loss = metrics["val"]
            save(
                {
                    "model": student.state_dict(),
                    "epoch": epoch,
                    "val_loss": best_loss,
                },
                best,
            )

        scheduler.step()
        save(
            {
                "model": student.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "best_loss": best_loss,
                "history": history,
                "loader_rng": dataloaders["train"].generator.get_state(),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all(),
            },
            last,
        )

        print(
            f"Epoch {epoch:02d}/22 | "
            f"Train {metrics['train']:.4f} | "
            f"Val {metrics['val']:.4f} | "
            f"Best {best_loss:.4f}"
        )

    report = pd.DataFrame(history)
    if len(report) == 10:
        csv = out / "student_22_history.csv"
        if not csv.exists():
            report.to_csv(csv, index=False)

        best_epoch = int(report.loc[report["val"].idxmin(), "epoch"])
        fig, ax = plt.subplots(figsize=(7, 3.5))
        for phase in ("train", "val"):
            ax.plot(report["epoch"], report[phase], "o-", label=phase.title())
        student_12_best_val = torch.load(base, map_location="cpu", weights_only=True)[
            "val_loss"
        ]
        ax.axhline(
            student_12_best_val,
            color="gray",
            linestyle="--",
            label="Student-12 best validation",
        )
        ax.set(
            xlabel="Epoch",
            ylabel="Distillation loss",
            title="Student-22 · Continued training",
        )
        ax.legend()
        ax.grid(alpha=0.2)
        fig.tight_layout()

        plot = out / "student_22_loss.png"
        if not plot.exists():
            fig.savefig(plot, dpi=160)
        plt.show()
        plt.close(fig)

        print(f"\nBest epoch: {best_epoch}/22")
        print(f"Best validation loss: {best_loss:.4f}")
        print("Checkpoint:", best)
        print("History:", csv)

    return report
