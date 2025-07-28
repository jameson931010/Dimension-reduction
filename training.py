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
KFOLDS = 4 # KFold cross validation
VAL_RATIO = 0.1 # 10% training data for validation
EPOCHS = 30
PATIENCE = 5
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = EMG128Dataset(dataset_dir="/tmp2/b12902141/DR/CapgMyo-DB-a", window_size=100)
# --------------------------

def training(train_loader, val_loader, fold):
    torch.manual_seed(fold)
    model = EMG128CAE().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    best_val_loss = float('inf')
    best_model_state = None
    no_improve_cnt = 0

    for epoch in range(1, EPOCHS+1):
        # Training
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"{fold} fold, Epoch {epoch}/{EPOCHS}, training")

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
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {epoch}/{EPOCHS}, validating")
            for batch, _ in pbar:
                batch = batch.to(DEVICE)
                outputs = model(batch)
                loss = criterion(outputs, batch)

                val_loss += loss.item()
                pbar.set_postfix(loss=loss.item())

        print(f"Epoch [{epoch}/{EPOCHS}], Training Loss: {train_loss/len(train_loader):.6f}, Validation Loss: {val_loss/len(val_loader):.6f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break
    torch.save(best_model_state, f"cae_fold{fold}.pth")
    print(f"Saved model for {fold} fold")


kf = KFold(n_splits=KFOLDS, shuffle=True, random_state=141)
for fold, (indeces, _) in enumerate(kf.split(dataset), start=1):
    print(f"\nFold {fold}/{KFOLDS}")
    val_split = int(len(indeces) * VAL_RATIO)
    val_idx = indeces[:val_split]
    train_idx = indeces[val_split:]

    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=BATCH_SIZE)

    training(train_loader, val_loader, fold)
