import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F
from sklearn.model_selection import KFold
from model import EMG128CAE
from dataset import EMG128Dataset
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# --------- Config ---------
DATA_PATH = "/tmp2/b12902141/DR/CapgMyo-DB-a"
WINDOW_SIZE = 100
KFOLDS = 4 # KFold cross validation
BATCH_SIZE = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# --------------------------

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

def plot(original, recon, title):
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--heatmap', action='store_true')
    parser.add_argument('name')
    args = parser.parse_args()
    if not args.name:
        print("Usage: python3 evaluation.py testing_name")
        exit(1)
    dataset = EMG128Dataset(dataset_dir=DATA_PATH, window_size=WINDOW_SIZE, subject_list=[1])
    kf = KFold(n_splits=KFOLDS, shuffle=True, random_state=141)

    for fold, (train, indeces) in enumerate(kf.split(dataset), start=1):
        if fold ==3:
            break
        print(f"\nFold {fold}/{KFOLDS}")

        # Initialization
        total_mse = 0.0
        total_power = 0.0
        total_snr = 0.0
        total_sample = 0

        train_mse = 0.0
        train_power = 0.0
        train_snr = 0.0
        train_sample = 0

        original_size = 0
        compressed_size = 0
        plotted = False

        train_loader = DataLoader(Subset(dataset, train[int(len(train)*0.1):]), batch_size=BATCH_SIZE)
        test_loader = DataLoader(Subset(dataset, indeces), batch_size=BATCH_SIZE)
        model = EMG128CAE(num_pooling=2, num_filter=4).to(DEVICE)
        model.load_state_dict(torch.load(f"cae_fold{fold}.pth", map_location=DEVICE))
        model.eval()

        with torch.no_grad():
            pbar = tqdm(train_loader)
            for batch, _ in pbar:
                # Visualize result
                if not plotted:
                    plotted = True
                    original = batch[0].squeeze().numpy()
                    reconstructed = model(batch[0].unsqueeze(0).to(DEVICE)).squeeze().cpu().numpy()
                    prd = 100 * np.sqrt(np.sum((original - reconstructed) ** 2) / np.sum(original ** 2))
                    with open(f"result/train_{args.name}", 'a') as f:
                        f.write(f"PRD for sample: {prd:.2f}%\n")
                    plot(original, reconstructed, f"{args.name}_train_{fold}")
                    if not args.heatmap:
                        plot_heatmap(original, reconstructed, f"{args.name}_train_{fold}")

                batch = batch.to(DEVICE)  # shape: (batch_size, 1, 100, 128)
                recon = model(batch)
                batch_size = batch.shape[0]

                # MSE
                mse = F.mse_loss(recon, batch, reduction='sum').item()
                train_mse += mse

                # PRD
                power = torch.sum(batch ** 2).item()
                train_power += power

                # SNR
                train_snr += 10 * np.log10(power / (mse + 1e-8)) * batch_size
                train_sample += batch_size
            plotted = False
            pbar = tqdm(test_loader)
            for batch, _ in pbar:
                # Visualize result
                if not plotted:
                    plotted = True
                    original = batch[0].squeeze().numpy()
                    reconstructed = model(batch[0].unsqueeze(0).to(DEVICE)).squeeze().cpu().numpy()
                    prd = 100 * np.sqrt(np.sum((original - reconstructed) ** 2) / np.sum(original ** 2))
                    with open(f"result/{args.name}", 'a') as f:
                        f.write(f"PRD for sample: {prd:.2f}%\n")
                    plot(original, reconstructed, f"{args.name}_{fold}")
                    if not args.heatmap:
                        plot_heatmap(original, reconstructed, f"{args.name}_{fold}")

                    """
                    # Test training set
                    original = dataset[train[0]][0].unsqueeze(0)
                    reconstructed = model(original.to(DEVICE)).squeeze().cpu().numpy()
                    original = original.squeeze().numpy()
                    prd = 100 * np.sqrt(np.sum((original - reconstructed) ** 2) / np.sum(original ** 2))
                    with open(f"result/{args.name}_train", 'a') as f:
                        f.write(f"PRD for sample: {prd:.2f}%\n")
                    plot(original, reconstructed, f"{args.name}_train_{fold}")
                    """

                batch = batch.to(DEVICE)  # shape: (batch_size, 1, 100, 128)
                recon = model(batch)
                batch_size = batch.shape[0]

                # MSE
                mse = F.mse_loss(recon, batch, reduction='sum').item()
                total_mse += mse

                # PRD
                power = torch.sum(batch ** 2).item()
                total_power += power

                # SNR
                total_snr += 10 * np.log10(power / (mse + 1e-8)) * batch_size

                # CR
                original_size += batch_size * 100 * 128 * 32  # 32-bit float
                compressed_size += model.encoder(batch).numel() * 32

                total_sample += batch_size

        # Metrics
        avg_mse = total_mse / (total_sample * 100 * 128)
        avg_snr = total_snr / total_sample
        prd = 100 * np.sqrt(total_mse / (total_power+1e-8))
        cr = original_size / compressed_size
        qs = cr / prd
        avg_mse_t = train_mse / (train_sample * 100 * 128)
        avg_snr_t = train_snr / train_sample
        prd_t = 100 * np.sqrt(train_mse / (train_power+1e-8))
        qs_t = cr / prd_t

        with open(f"result/{args.name}", 'a') as f:
            f.write(f"Evaluation Results for {fold} fold:\n")
            f.write(f"  Avg MSE: {avg_mse:.6f}\n")
            f.write(f"  Avg SNR: {avg_snr:.2f} dB\n")
            f.write(f"  PRD: {prd:.2f} %\n")
            f.write(f"  CR: {cr:.2f}\n")
            f.write(f"  QS: {qs:.4f}\n")
        print(f"Evaluation Results for {fold} fold:\n")
        print(f"  Avg MSE: {avg_mse:.6f}\n")
        print(f"  Avg SNR: {avg_snr:.2f} dB\n")
        print(f"  PRD: {prd:.2f} %\n")
        print(f"  CR: {cr:.2f}\n")
        print(f"  QS: {qs:.4f}\n")
        print(f"  Avg MSE for train: {avg_mse_t:.6f}\n")
        print(f"  Avg SNR for train: {avg_snr_t:.2f} dB\n")
        print(f"  PRD for train: {prd_t:.2f} %\n")
        print(f"  QS: {qs_t:.4f}\n")
        with open(f"result/train_{args.name}", 'a') as f:
            f.write(f"Evaluation Results for {fold} fold:\n")
            f.write(f"  Avg MSE: {avg_mse_t:.6f}\n")
            f.write(f"  Avg SNR: {avg_snr_t:.2f} dB\n")
            f.write(f"  PRD: {prd_t:.2f} %\n")
            f.write(f"  QS: {qs_t:.4f}\n")
