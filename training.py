import os
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from model import EMG128CAE
from dataset import EMG128Dataset

# --------- Config ---------
DATA_PATH = "/tmp2/b12902141/DR/CapgMyo-DB-a"
WINDOW_SIZE = 100
KFOLDS = 4 # KFold cross validation
VAL_RATIO = 0.1 # 10% training data for validation
EPOCHS = 20
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# --------------------------

# Initialization
dataset = EMG128Dataset(dataset_dir=DATA_PATH, window_size=WINDOW_SIZE)
model = EMG128CAE().to(DEVICE)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

def training(train_loader, val_loader):
    for epoch in range(1, EPOCHS+1):
        # Training
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}, training")

        for batch, _ in pbar:
            batch = batch.to(DEVICE) # batch: (BATCH_SIZE, 1 channel for convolution, 100, 128)
            optimizer.zero_grad()
            outputs = model(batch)
            loss = criterion(outputs, batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        # Validation
        model.eval()
        val_loss = 0.0
        pbar = tqdm(val_loader, desc=f"Epoch {epoch}/{EPOCHS}, validating")
        for batch, _ in pbar:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(batch)
            loss = criterion(outputs, batch)
            loss.backward()
            optimizer.step()

            val_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        print(f"Epoch [{epoch}/{EPOCHS}], Training Loss: {train_loss/len(train_loader):.6f}, Validation Loss: {val_loss/len(val_loader):.6f}")


kf = KFold(n_splits=KFOLDS, shuffle=True, random_state=141)
for fold, (indeces, _) in enumerate(kf.split(dataset), start=1):
    print(f"\nFold {fold}/{KFOLDS}")
    val_split = int(len(indeces) * VAL_RATIO)
    val_idx = indeces[:val_split]
    train_idx = indeces[val_split:]

    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=BATCH_SIZE)

    training(train_loader, val_loader)
    # Save model per fold
    #out_name = f"cae_{os.path.basename(subject_path)}_fold{fold}.pth"
    out_name = f"cae_fold{fold}.pth"
    torch.save(model.state_dict(), out_name)
    print(f"Saved model: {out_name}")
