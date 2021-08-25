import torch
import os

def save_checkpoint(
    epoch,
    acc,
    model,
    optimizer,
    # scheduler,
    # scaler,
    path
    ):
    checkpoint = {
        "epoch": epoch,
        "acc": acc,
        "model_state_dict": model.state_dict(),
        "optim_state_dict": optimizer.state_dict(),
        # "sched_state_dict": scheduler.state_dict(),
        # "scaler_state_dict": scaler.state_dict(),
        }
    torch.save(checkpoint, os.path.join(path, "epoch_" + str(epoch + 1) + ".pth"))
