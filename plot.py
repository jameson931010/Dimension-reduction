import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def plot_heatmap(original, reconstructed, title):
    fig, axs = plt.subplots(1, 3, figsize=(14, 6))
    vmax = max(original.max(), reconstructed.max())
    vmin = min(original.min(), reconstructed.min())

    sns.heatmap(original.T, ax=axs[0], cmap='viridis', cbar=True, vmin=vmin, vmax=vmax)
    axs[0].set_title('Original Signal')
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('Channels')

    sns.heatmap(reconstructed.T, ax=axs[1], cmap='viridis', cbar=True, vmin=vmin, vmax=vmax)
    axs[1].set_title('Reconstructed Signal')
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('Channels')

    error = np.abs(original - reconstructed)
    sns.heatmap(error.T, ax=axs[2], cmap='Reds', cbar=True)
    axs[2].set_title('Absolute Error')

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(f"visual/heatmap_{title}.png")
    plt.close()

def plot_channel(original, reconstructed, title):
    CHANNELS_TO_PLOT = [0, 1, 2, 6, 7, 8, 9, 10, 14, 15, 16, 17]
    plt.figure(figsize=(14, 2 * len(CHANNELS_TO_PLOT)))

    for i, ch in enumerate(CHANNELS_TO_PLOT):
        plt.subplot(len(CHANNELS_TO_PLOT), 1, i+1)
        plt.plot(original[:, ch], label="Original", color="black", linewidth=1)
        plt.plot(reconstructed[:, ch], label="Reconstructed", color="red", linestyle='--', linewidth=1)
        plt.title(f"Channel {ch}")
        plt.xlabel("Time (samples)")
        plt.ylabel("Amplitude")
        plt.grid(True)
        if i == 0:
            plt.legend()

    plt.tight_layout()
    plt.suptitle(title, y=1.02, fontsize=16)
    # plt.show()
    plt.savefig(f"visual/{title}.png")
    plt.close()
