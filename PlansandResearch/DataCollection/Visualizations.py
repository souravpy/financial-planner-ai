import matplotlib.pyplot as plt

class Visualizer:
    def plot_loss_plt(
        self, folder_name: str, file_name: str, history, ymin: int = 0, ymax: int = 10000
    ):
        """Plot loss"""
        plt.plot(history.history["loss"], label="loss")
        plt.plot(history.history["val_loss"], label="val_loss")
        plt.ylim([ymin, ymax])
        plt.xlabel("Epoch")
        plt.ylabel("Error")
        plt.legend()
        plt.grid(True)
        plt.show()
        plt.savefig(f"{folder_name}{file_name}_loss.png")
        raise NotImplementedError