import math
import pandas as pd
import matplotlib.pyplot as plt
import torch
from tqdm.auto import tqdm


def train_student_12(student, dataloaders, criterion, optimizer, config):

    out = config["output_dir"] / "training" / "seed42"
    out.mkdir(parents=True, exist_ok=True)
    best_path = out / "student_12_best.pt"
    last_path = out / "student_12_last.pt"

    epochs = 12
    device = config["device"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    def save_checkpoint(state, path):
        temp = path.with_suffix(".tmp")
        torch.save(state, temp)
        temp.replace(path)

    history, best_val, start_epoch = [], float("inf"), 1

    if last_path.exists():
        state = torch.load(last_path, map_location="cpu", weights_only=True)
        student.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        dataloaders["train"].generator.set_state(state["loader_rng"])
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])
        history = state["history"]
        best_val = state["best_val"]
        start_epoch = state["epoch"] + 1
        print(f"Resuming from completed epoch {start_epoch - 1}")
    elif best_path.exists():
        raise FileExistsError("Existing best checkpoint without resume state.")

    for epoch in range(start_epoch, epochs + 1):
        losses = {}

        for phase in ("train", "val"):
            training = phase == "train"
            student.train(training)
            total_loss, total_samples = 0.0, 0

            for images, targets in tqdm(
                dataloaders[phase],
                desc=f"Epoch {epoch:02d}/{epochs} · {phase}",
                leave=False,
            ):
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

                if training:
                    optimizer.zero_grad(set_to_none=True)

                with torch.set_grad_enabled(training):
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        predictions = student(images)

                    loss = criterion(predictions.float(), targets)

                    if not math.isfinite(loss.item()):
                        raise FloatingPointError(
                            f"Non-finite {phase} loss at epoch {epoch}"
                        )

                    if training:
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(
                            student.parameters(), max_norm=1.0
                        )
                        optimizer.step()

                total_loss += loss.item() * len(images)
                total_samples += len(images)

            losses[phase] = total_loss / total_samples

        history.append(
            {
                "epoch": epoch,
                "train": losses["train"],
                "val": losses["val"],
                "backbone_lr": optimizer.param_groups[0]["lr"],
                "head_lr": optimizer.param_groups[1]["lr"],
            }
        )

        if losses["val"] < best_val:
            best_val = losses["val"]
            save_checkpoint(
                {
                    "model": student.state_dict(),
                    "epoch": epoch,
                    "val_loss": best_val,
                },
                best_path,
            )

        scheduler.step()

        save_checkpoint(
            {
                "model": student.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "best_val": best_val,
                "history": history,
                "loader_rng": dataloaders["train"].generator.get_state(),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all(),
            },
            last_path,
        )

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"Train {losses['train']:.4f} | "
            f"Val {losses['val']:.4f} | "
            f"Best {best_val:.4f}"
        )

    report = pd.DataFrame(history)

    if len(report) == epochs:
        csv_path = out / "student_12_history.csv"
        if not csv_path.exists():
            report.to_csv(csv_path, index=False)

        best_epoch = int(report.loc[report["val"].idxmin(), "epoch"])

        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.plot(report["epoch"], report["train"], label="Train", marker="o")
        ax.plot(report["epoch"], report["val"], label="Validation", marker="o")
        ax.axvline(
            best_epoch, linestyle="--", color="gray", label=f"Best epoch {best_epoch}"
        )
        ax.set(
            xlabel="Epoch",
            ylabel="Distillation loss",
            title="Student-12 · Training convergence",
        )
        ax.legend()
        ax.grid(alpha=0.2)
        fig.tight_layout()

        plot_path = out / "student_12_loss.png"
        if not plot_path.exists():
            fig.savefig(plot_path, dpi=160)
        plt.show()
        plt.close(fig)

        print(f"\nBest epoch: {best_epoch}/{epochs}")
        print(f"Best validation loss: {best_val:.4f}")
        print("Best checkpoint:", best_path)
        print("History:", csv_path)

    return report
