import sys
import os
import random
import torch
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from sklearn.model_selection import KFold
from tqdm import tqdm
from model import EMG128CAE
from dataset import EMG128Dataset, REPETITION, SAMPLE_LEN
from plot import plot_channel, plot_heatmap, plot_metric

# --------- Config ---------
KFOLDS = 5 # KFold cross validation
VAL_RATIO = 0.1 # 10% training data for validation
EPOCHS = 500
PATIENCE = 25
BATCH_SIZE = 10 # Should be a multiple of the number of window within a .mat file 
CRITERION = nn.SmoothL1Loss() # nn.L1Loss() # nn.MSELoss(), nn.SmoothL1Loss()
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-5

NUM_POOLING = 3
NUM_FILTER = 2

random.seed(141)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

WINDOW_SIZE = 100
SUBJECT_LIST = [1] #[x for x in range(1, 19)]
FIRST_N_GESTURE = 8
NAME = f"{sys.argv[1]}_e{EPOCHS}-{PATIENCE}_b{BATCH_SIZE}_lr{LEARNING_RATE}_wd{WEIGHT_DECAY}_{NUM_POOLING}_{NUM_FILTER}"
dataset = EMG128Dataset(dataset_dir="/tmp2/b12902141/DR/CapgMyo-DB-a", window_size=WINDOW_SIZE, subject_list=SUBJECT_LIST, first_n_gesture=FIRST_N_GESTURE)
EARLY_STOPPING = True
NOT_ALL_KFOLD = True
PRINT_TRAIN = True
# --------------------------

def process_one_fold(train_idx, val_idx, test_idx, fold):
    torch.manual_seed(fold)
    model = EMG128CAE(num_pooling=NUM_POOLING, num_filter=NUM_FILTER).to(DEVICE)

    # Training
    dual_print(f"Fold:{fold} training")
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=BATCH_SIZE, shuffle=False)
    model_state = training(model, train_loader, val_loader)

    torch.save(model_state, f"cae_fold{fold}.pth")
    with open(f"log/{sys.argv[1]}.log", 'a') as f:
        f.write(f"Model saved for {fold} fold\n")

    # Evaluation
    ordered_train_loader = DataLoader(Subset(dataset, train_idx), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(Subset(dataset, test_idx), batch_size=BATCH_SIZE, shuffle=False)
    if PRINT_TRAIN:
        evaluation(model, ordered_train_loader, name=f"{NAME}_train_{fold}")
    evaluation(model, test_loader, name=f"{NAME}_test_{fold}")
    

def training(model, train_loader, val_loader):
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
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}, training")

        for batch in pbar:
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
            for batch in val_loader:
                batch = batch.to(DEVICE)
                outputs = model(batch)
                loss = criterion(outputs, batch)
                val_loss += loss.item()

        #scheduler.step(val_loss/len(val_loader))
        with open(f"log/{sys.argv[1]}.log", 'a') as f:
            f.write(f"Epoch [{epoch}/{EPOCHS}], Training Loss: {train_loss/len(train_loader):.6f}, Validation Loss: {val_loss/len(val_loader):.6f}\n")

        # Early stopping
        if EARLY_STOPPING:
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
    if EARLY_STOPPING:
        return best_model_state
    else:
        return model.state_dict()

def evaluation(model, data_loader, name, plot=True):
    dual_print(f"Evaluating {name}")
    model.eval()
    with torch.no_grad():
        batch = next(iter(data_loader)).to(DEVICE)
        CR = batch.shape[0] / model.encoder(batch).numel()
        if plot:
            original = batch[0].squeeze().cpu().numpy()
            reconstructed = model(batch[0].unsqueeze(0)).squeeze().cpu().numpy()
            prd = 100 * np.sqrt(np.sum((original - reconstructed) ** 2) / np.sum(original ** 2))
            dual_print(f"PRD for sample: {prd:.3f}")
            plot_channel(original, reconstructed, name)
            plot_heatmap(original, reconstructed, name)

        PRDN = np.empty(len(data_loader))
        SNR = np.empty(len(data_loader))
        QS = np.empty(len(data_loader))
        cnt = 0
        for batch in data_loader:
            batch = batch.to(DEVICE)  # shape: (dataset.window_num, 1, 100, 128)
            recon = model(batch)
            SE = torch.sum((batch-recon) ** 2)
            SS = torch.sum(batch ** 2)
            # The following assum mean=0
            prdn = torch.sqrt(SE/SS).item() * 100
            snr = 10 * torch.log10(SS/SE).item()
            # qs = CR / prdn
            PRDN[cnt] = prdn
            SNR[cnt] = snr
            #QS[cnt] = qs
            cnt += 1

        dual_print(f"   PRDN: {PRDN.mean():.3f}")
        dual_print(f"   SNR: {SNR.mean():.3f}")
        #dual_print(f"   QS: {QS.mean():.3f}")
        if plot:
            if len(SUBJECT_LIST) == 1:
                plot_metric(PRDN.reshape(FIRST_N_GESTURE, -1), xlabel="repetition", ylabel="gesture", title=f"PRDN_{name}")
                plot_metric(SNR.reshape(FIRST_N_GESTURE, -1), xlabel="repetition", ylabel="gesture", title=f"SNR_{name}")
                #plot_metric(QS.reshape(FIRST_N_GESTURE, -1), xlabel="repetition", ylabel="gesture", title=f"QS_{name}")
            else:
                plot_metric(PRDN.reshape(len(SUBJECT_LIST), FIRST_N_GESTURE, -1).mean(axis=2), xlabel="gesture", ylabel="subject", title=f"PRDN_{name}")
                plot_metric(SNR.reshape(len(SUBJECT_LIST), FIRST_N_GESTURE, -1).mean(axis=2), xlabel="gesture", ylabel="subject", title=f"SNR_{name}")
                #plot_metric(QS.reshape(len(SUBJECT_LIST), FIRST_N_GESTURE, -1).mean(axis=2), xlabel="gesture", ylabel="subject", title=f"QS_{name}")


def check_dimension():
    model = EMG128CAE(num_pooling=NUM_POOLING, num_filter=NUM_FILTER).to(DEVICE)
    x = torch.randn(1, 1, 100, 128).to(DEVICE)
    for i, layer in enumerate(model.encoder):
        x = layer(x)
        print(f"shape of {i}: {x.shape}")
    for i, layer in enumerate(model.decoder):
        x = layer(x)
        print(f"shape of {i}: {x.shape}")
    exit(0)

def dual_print(mes):
    print(mes)
    with open(f"log/{NAME}.log", 'a') as f:
        f.write(str(mes) + '\n')
    

if __name__ == '__main__':
    if len(sys.argv) <= 1:
        print("Usage: python3 training.py testing_name")
        exit(1)

    for fold in range(1, KFOLDS+1):
        if fold==3 and NOT_ALL_KFOLD:
            break
        with open(f"log/{sys.argv[1]}.log", 'a') as f:
            f.write(f"\nFold {fold}/{KFOLDS}\n")
        
        test_idx = []
        val_idx = []
        train_idx = []
        # non-inclusive upperbound of index of repetition for testing and validation
        test_bound = REPETITION//KFOLDS # 2
        val_bound = test_bound + int(REPETITION*VAL_RATIO) # 3

        # Sample repetition for testing
        # idx: (len(SUBJECT_LIST), FIRST_N_GESTURE * REPETITION * dataset.window_num)
        rep_sample_pool = range(REPETITION) # The pool of repetition to sample from
        for subject in SUBJECT_LIST:
            for gesture in range(FIRST_N_GESTURE):
                indeces = random.sample(rep_sample_pool, REPETITION)
                for rep in range(REPETITION):
                    idx_base = subject*dataset.subject_len + (gesture*REPETITION + rep)*dataset.window_num
                    if rep < test_bound:
                        test_idx.extend([idx_base + i for i in range(dataset.window_num)])
                    elif rep < val_bound:
                        val_idx.extend([idx_base + i for i in range(dataset.window_num)])
                    else:
                        train_idx.extend([idx_base + i for i in range(dataset.window_num)])

        process_one_fold(train_idx, val_idx, test_idx, fold)
