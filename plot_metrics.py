import matplotlib.pyplot as plt
import json
import os

def plot_training_curves(metrics_file="training_metrics.json", save_path="metrics_plot.png"):
    """Plot and save loss and accuracy curves from a saved metrics JSON file."""
    if not os.path.exists(metrics_file):
        print(f"❌ Metrics file not found: {metrics_file}")
        return

    with open(metrics_file, "r") as f:
        metrics = json.load(f)

    epochs = range(1, len(metrics["train_loss"]) + 1)

    plt.figure(figsize=(10, 4))

    # --- Loss ---
    plt.subplot(1, 2, 1)
    plt.plot(epochs, metrics["train_loss"], label="Train Loss", marker="o", color="tab:red")
    plt.plot(epochs, metrics["val_loss"], label="Val Loss", marker="o", color="tab:orange")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)

    # --- Accuracy ---
    plt.subplot(1, 2, 2)
    plt.plot(epochs, metrics["train_acc"], label="Train Acc", marker="o", color="tab:blue")
    plt.plot(epochs, metrics["val_acc"], label="Val Acc", marker="o", color="tab:green")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()

    print(f"✅ Plot saved as: {save_path}")


if __name__ == "__main__":
    plot_training_curves()

