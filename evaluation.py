import os
import torch
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F
from sklearn.model_selection import KFold
from model import EMG128CAE
from dataset import EMG128Dataset
import numpy as np
from tqdm import tqdm

# --------- Config ---------
DATA_PATH = "/tmp2/b12902141/DR/CapgMyo-DB-a"
WINDOW_SIZE = 100
KFOLDS = 4 # KFold cross validation
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# --------------------------

dataset = EMG128Dataset(dataset_dir=DATA_PATH, window_size=WINDOW_SIZE)
kf = KFold(n_splits=KFOLDS, shuffle=True, random_state=141)

for fold, (_, indeces) in enumerate(kf.split(dataset), start=1):
    print(f"\nFold {fold}/{KFOLDS}")

    # Initialization
    total_mse = 0.0
    total_power = 0.0
    total_snr = 0.0
    total_sample = 0
    original_size = 0
    compressed_size = 0

    test_loader = DataLoader(Subset(dataset, indeces), batch_size=BATCH_SIZE)
    model = EMG128CAE().to(DEVICE)
    model.load_state_dict(torch.load(f"cae_fold{fold}.pth", map_location=DEVICE))
    model.eval()

    with torch.no_grad():
        pbar = tqdm(test_loader)
        for batch, _ in pbar:
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

    print(f"Evaluation Results:")
    print(f"  Avg MSE: {avg_mse:.6f}")
    print(f"  Avg SNR: {avg_snr:.2f} dB")
    print(f"  PRD: {prd:.2f} %")
    print(f"  CR: {cr:.2f}")
    print(f"  QS: {qs:.4f}")

