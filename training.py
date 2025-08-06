import sys
import os
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from full_model import EMG128CAE
from dataset import EMG128Dataset

# --------- Config ---------
KFOLDS = 4 # KFold cross validation
VAL_RATIO = 0.1 # 10% training data for validation
EPOCHS = 500
PATIENCE = 50
BATCH_SIZE = 4
CRITERION = nn.SmoothL1Loss() # nn.L1Loss() # nn.MSELoss(), nn.SmoothL1Loss()
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = EMG128Dataset(dataset_dir="/tmp2/b12902141/DR/CapgMyo-DB-a", window_size=100, subject_list=[1])
# --------------------------

def training(train_loader, val_loader, fold):
    torch.manual_seed(fold)
    model = EMG128CAE(num_pooling=4, num_filter=2).to(DEVICE)
    criterion = CRITERION
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)#, weight_decay=WEIGHT_DECAY)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
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

        #scheduler.step(val_loss/len(val_loader))
        with open(f"log/{sys.argv[1]}.log", 'a') as f:
            f.write(f"Epoch [{epoch}/{EPOCHS}], Training Loss: {train_loss/len(train_loader):.6f}, Validation Loss: {val_loss/len(val_loader):.6f}\n")

        # Early stopping
        """
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= PATIENCE:
                with open(f"log/{sys.argv[1]}.log", 'a') as f:
                    f.write(f"Early stopping at epoch {epoch}\n")
                break
        """
    best_model_state = model.state_dict()
    torch.save(best_model_state, f"cae_fold{fold}.pth")
    with open(f"log/{sys.argv[1]}.log", 'a') as f:
        f.write(f"Saved model for {fold} fold\n")

"""
model = EMG128CAE(num_pooling=3, num_filter=2).to(DEVICE)
x = torch.randn(1, 1, 100, 8, 16).to(DEVICE)
for i, layer in enumerate(model.encoder):
    x = layer(x)
    print(f"shape of {i}: {x.shape}")
for i, layer in enumerate(model.decoder):
    x = layer(x)
    print(f"shape of {i}: {x.shape}")
exit(0)
"""

if __name__ == '__main__':
    if len(sys.argv) <= 1:
        print("Usage: python3 training.py testing_name")
        exit(1)
    kf = KFold(n_splits=KFOLDS, shuffle=True, random_state=141)
    for fold, (indeces, _) in enumerate(kf.split(dataset), start=1):
        if fold==3:
            break
        with open(f"log/{sys.argv[1]}.log", 'a') as f:
            f.write(f"\nFold {fold}/{KFOLDS}\n")
        val_split = int(len(indeces) * VAL_RATIO)
        val_idx = indeces[:val_split]
        train_idx = indeces[val_split:]

        train_loader = DataLoader(Subset(dataset, train_idx), batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(Subset(dataset, val_idx), batch_size=BATCH_SIZE)

        training(train_loader, val_loader, fold)
