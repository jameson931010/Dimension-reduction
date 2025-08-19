import sys
import os
import random
import torch
import copy
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from sklearn.model_selection import KFold
from tqdm import tqdm
from vcae_model import EMG128CAE, INPUT_TIME_DIM, INPUT_CHANNEL_DIM
from diffusion_model import LatentDenoiser, LatentDDPM
from dataset import EMG128Dataset, LatentDataset, REPETITION, SAMPLE_LEN
from plot import plot_channel, plot_heatmap, plot_metric

# --------- Config ---------
# Training settings
KFOLDS = 5 # KFold cross validation
VAL_RATIO = 0.1 # 10% training data for validation
EPOCHS = 1600
PATIENCE = 40
BATCH_SIZE = 10 # Should be a multiple of the number of window within a .mat file 
BETA = 1
BETA_WARM_UP = 30
CRITERION = nn.MSELoss() # nn.L1Loss() nn.MSELoss(), nn.SmoothL1Loss()
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0 #1e-4

random.seed(141)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CAE settings
NUM_POOLING = int(sys.argv[2])
NUM_FILTER = int(sys.argv[3])
OUTPUT_TIME_DIM = INPUT_TIME_DIM // pow(2, NUM_POOLING)
OUTPUT_CHANNEL_DIM = INPUT_CHANNEL_DIM // pow(2, NUM_POOLING)

# Diffusion settings
DIFF_T = 400
DIFF_LR = 2e-4
DIFF_EPOCHS = 400
DIFF_BETA_START = 1e-4
DIFF_BETA_END = 2e-2
DIFF_SAMPLES_TO_SAVE = 2

# Config for experiment
WINDOW_SIZE = 100
SUBJECT_LIST = [int(sys.argv[4])] # [x for x in range(1, 19)]
FIRST_N_GESTURE = 8
NAME = f"{sys.argv[1]}_e{EPOCHS}-{PATIENCE}_b{BATCH_SIZE}_lr{LEARNING_RATE}_wd{WEIGHT_DECAY}_{NUM_POOLING}_{NUM_FILTER}"
dataset = EMG128Dataset(dataset_dir="/tmp2/b12902141/DR/CapgMyo-DB-a", window_size=WINDOW_SIZE, subject_list=SUBJECT_LIST, first_n_gesture=FIRST_N_GESTURE)
EARLY_STOPPING = True
DIFF_EARLY_STOPPING = True
NOT_ALL_KFOLD = True
PRINT_TRAIN = True
PLOT_METRIC = False
# --------------------------

def process_one_fold(train_idx, val_idx, test_idx, fold, train=False, train_ddpm=True):
    torch.manual_seed(fold)
    vcae_model = EMG128CAE(num_pooling=NUM_POOLING, num_filter=NUM_FILTER).to(DEVICE)
    denoiser = LatentDenoiser(in_ch=NUM_FILTER).to(DEVICE)
    ddpm = LatentDDPM(denoiser, T=DIFF_T, beta_start=DIFF_BETA_START, beta_end=DIFF_BETA_END, device=DEVICE).to(DEVICE)

    # Training vcae model
    dual_print(f"Fold:{fold} training")
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=BATCH_SIZE, shuffle=False)
    if train:
        vcae_state = training(vcae_model, train_loader, val_loader)
        torch.save(vcae_state, f"model/cae_for_diff{fold}.pth")
        with open(f"log/{sys.argv[1]}.log", 'a') as f:
            f.write(f"Model saved for {fold} fold\n")
    else:
        vcae_state = torch.load(f"model/cae_for_diff{fold}.pth")
    vcae_model.load_state_dict(vcae_state)

    # Freeze the model
    vcae_model.eval()
    for p in vcae_model.parameters():
        p.requires_grad = False

    # Generate dataset for diffusion model
    ddpm_train_set = LatentDataset(Subset(dataset, train_idx), vcae_model, DEVICE)
    ddpm_val_set = LatentDataset(Subset(dataset, val_idx), vcae_model, DEVICE)
    ddpm_test_set = LatentDataset(Subset(dataset, test_idx), vcae_model, DEVICE)

    # Training diffusion model
    ddpm_train_loader = DataLoader(ddpm_train_set, batch_size=BATCH_SIZE, shuffle=True)
    ddpm_val_loader = DataLoader(ddpm_val_set, batch_size=BATCH_SIZE, shuffle=False)
    if train_ddpm:
        ddpm_state = training_diffusion(denoiser, ddpm, ddpm_train_loader, ddpm_val_loader, fold)
        torch.save(ddpm_state, f"model/ddpm_fold{fold}.pth")
        with open(f"log/{sys.argv[1]}.log", 'a') as f:
            f.write(f"DDPM model saved for {fold} fold\n")
    else:
        ddpm_state = torch.load(f"model/ddpm_fold{fold}.pth")
    denoiser.load_state_dict(ddpm_state)

    # Evaluation
    ordered_train_loader = DataLoader(Subset(dataset, train_idx), batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(Subset(dataset, test_idx), batch_size=BATCH_SIZE, shuffle=False)
    if PRINT_TRAIN:
        evaluation(vcae_model, ordered_train_loader, name=f"{NAME}_train_{fold}")
    evaluation(vcae_model, test_loader, name=f"{NAME}_test_{fold}")

    # Evaluate_ddpm
    #ddpm_ordered_train_loader = DataLoader(ddpm_train_set, batch_size=BATCH_SIZE, shuffle=False)
    #ddpm_test_loader = DataLoader(ddpm_test_set, batch_size=BATCH_SIZE, shuffle=False)
    if PRINT_TRAIN:
        check_ddpm(vcae_model, denoiser, ddpm, ordered_train_loader, name=f"ddpm_{NAME}_train_{fold}")
    check_ddpm(vcae_model, denoiser, ddpm, test_loader, name=f"ddpm_{NAME}_test_{fold}")

    

def training(model, train_loader, val_loader):
    #criterion = CRITERION
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)#, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    best_val_loss = float('inf')
    best_model_state = None
    no_improve_epochs = 0

    for epoch in range(1, EPOCHS+1):
        # Training
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}, training")
        beta = min(BETA, epoch / BETA_WARM_UP * BETA)

        for batch in pbar:
            batch = batch.to(DEVICE) # batch: (BATCH_SIZE, 1 channel for convolution, 100, 128)
            optimizer.zero_grad()
            #outputs = model(batch)
            outputs, mu, logvar = model(batch)

            #loss = criterion(outputs, batch)
            loss = cal_loss(batch, outputs, mu, logvar, beta)
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
                #outputs = model(batch)
                outputs, mu, logvar = model(batch)

                #loss = criterion(outputs, batch)
                beta = min(1.0, epoch / BETA_WARM_UP)
                loss = cal_loss(batch, outputs, mu, logvar, beta)
                val_loss += loss.item()

        #scheduler.step(val_loss/len(val_loader))
        with open(f"log/{sys.argv[1]}.log", 'a') as f:
            f.write(f"Epoch [{epoch}/{EPOCHS}], Training Loss: {train_loss/len(train_loader):.6f}, Validation Loss: {val_loss/len(val_loader):.6f}\n")

        # Early stopping
        if EARLY_STOPPING:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = copy.deepcopy(model.state_dict())
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

def training_diffusion(denoiser, ddpm, train_loader, val_loader, fold):
    optimizer = optim.Adam(denoiser.parameters(), lr=DIFF_LR)

    best_loss = float('inf')
    best_model_state = None
    no_improve_epochs = 0

    for epoch in range(1, DIFF_EPOCHS+1):
        denoiser.train()
        # Training
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"DDPM epoch {epoch}/{DIFF_EPOCHS}")
        for mu in pbar:
            mu = mu.to(DEVICE)
            optimizer.zero_grad()
            loss = ddpm.training_loss(mu)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        # Validation
        ddpm.denoiser.eval()
        val_loss = 0.0
        with torch.no_grad():
            for mu in val_loader:
                mu = mu.to(DEVICE)
                loss = ddpm.training_loss(mu)
                val_loss += loss.item()

        #scheduler.step(val_loss/len(val_loader))
        with open(f"log/{sys.argv[1]}.log", 'a') as f:
            f.write(f"[DDPM] Epoch [{epoch}/{EPOCHS}], Training Loss: {train_loss/len(train_loader):.6f}, Validation Loss: {val_loss/len(val_loader):.6f}\n")

        # Early stopping
        if DIFF_EARLY_STOPPING:
            if val_loss < best_loss:
                best_loss = val_loss
                no_improve_epochs = 0
                best_model_state = copy.deepcopy(denoiser.state_dict())
            else:
                no_improve_epochs += 1
                if no_improve_epochs >= PATIENCE:
                    with open(f"log/{sys.argv[1]}.log", 'a') as f:
                        f.write(f"[DDPM] Early stopping at epoch {epoch}\n")
                    break
    if EARLY_STOPPING:
        return best_model_state
    else:
        return denoiser.state_dict()

def evaluation(model, data_loader, name, plot=False):
    dual_print(f"Evaluating {name}")
    model.eval()
    with torch.no_grad():
        batch = next(iter(data_loader)).to(DEVICE)
        # Be careful when calling the encoder and decoder individually, which may cause memory leakage or incorrect result
        code, mu, logvar = model.encode(batch)

        #code_size = model.encode(batch)[0].numel()
        code_size = code[0].numel()
        CR = INPUT_TIME_DIM * INPUT_CHANNEL_DIM / code_size # Not considering quantization
        #CR = batch.shape[0] / model.encoder(batch).numel()
        #CR = batch.shape[0] / model.encoder(batch.squeeze(1).permute(0, 2, 1)).numel()
        if plot:
            #original = batch[0].squeeze().cpu().numpy()
            #reconstructed = model(batch[0].unsqueeze(0)).squeeze().cpu().numpy()
            original = batch[0].squeeze().cpu().numpy()
            reconstructed = model(batch[0].unsqueeze(0))[0].squeeze().cpu().numpy()
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
            recon, _, _ = model(batch)
            #recon = model(batch)
            SE = torch.sum((batch-recon) ** 2)
            SS = torch.sum(batch ** 2)
            # The following assum mean=0
            prdn = torch.sqrt(SE/SS).item() * 100
            snr = 10 * torch.log(SS/SE).item()
            # qs = CR / prdn
            PRDN[cnt] = prdn
            SNR[cnt] = snr
            #QS[cnt] = qs
            cnt += 1

        dual_print(f"   CR: {CR:.2f}")
        dual_print(f"   PRDN: {PRDN.mean():.3f}")
        dual_print(f"   SNR: {SNR.mean():.3f}")
        #dual_print(f"   QS: {QS.mean():.3f}")
        if PLOT_METRIC:
            if len(SUBJECT_LIST) == 1:
                plot_metric(PRDN.reshape(FIRST_N_GESTURE, -1), xlabel="repetition", ylabel="gesture", title=f"PRDN_{name}")
                plot_metric(SNR.reshape(FIRST_N_GESTURE, -1), xlabel="repetition", ylabel="gesture", title=f"SNR_{name}")
                #plot_metric(QS.reshape(FIRST_N_GESTURE, -1), xlabel="repetition", ylabel="gesture", title=f"QS_{name}")
            else:
                plot_metric(PRDN.reshape(len(SUBJECT_LIST), FIRST_N_GESTURE, -1).mean(axis=2), xlabel="gesture", ylabel="subject", title=f"PRDN_{name}")
                plot_metric(SNR.reshape(len(SUBJECT_LIST), FIRST_N_GESTURE, -1).mean(axis=2), xlabel="gesture", ylabel="subject", title=f"SNR_{name}")
                #plot_metric(QS.reshape(len(SUBJECT_LIST), FIRST_N_GESTURE, -1).mean(axis=2), xlabel="gesture", ylabel="subject", title=f"QS_{name}")

def check_ddpm(vcae_model, denoiser, ddpm, data_loader, name, denoise_step=50):
    denoiser.eval()
    with torch.no_grad():
        batch = next(iter(data_loader)).to(DEVICE)
        z_0, _, _ = vcae_model.encode(batch)
        # Pick a noise step denoise_step-1 and add forward noise
        t_0 = torch.full((z_0.size(0),), denoise_step-1, device=DEVICE, dtype=torch.long)
        x_k = ddpm.q_sample(z_0, t_0)                       # noisy latent at step denoise_step-1

        # Walk back to 0
        x_t = x_k
        for step in reversed(range(denoise_step)):
            t = torch.full((z_0.size(0),), step, device=DEVICE, dtype=torch.long)
            x_t = ddpm.p_sample_step(x_t, t)

        z_refined = x_t
        x_refined = vcae_model.decode(z_refined)
        #x_raw = vcae_model.decode(z_0)
        original = batch[0].squeeze().cpu().numpy()
        reconstructed = x_refined[0].squeeze().cpu().numpy()
        plot_channel(original, reconstructed, name)
        plot_heatmap(original, reconstructed, name)

        PRDN = np.empty(len(data_loader))
        SNR = np.empty(len(data_loader))
        QS = np.empty(len(data_loader))
        cnt = 0
        for batch in data_loader:
            batch = batch.to(DEVICE)  # shape: (dataset.window_num, 1, 100, 128)
            z_0, _, _ = vcae_model.encode(batch)
            t_0 = torch.full((z_0.size(0),), denoise_step-1, device=DEVICE, dtype=torch.long)
            x_k = ddpm.q_sample(z_0, t_0)                       # noisy latent at step denoise_step-1

            x_t = x_k
            for step in reversed(range(denoise_step)):
                t = torch.full((z_0.size(0),), step, device=DEVICE, dtype=torch.long)
                x_t = ddpm.p_sample_step(x_t, t)

            z_refined = x_t
            recon = vcae_model.decode(z_refined)
            SE = torch.sum((batch-recon) ** 2)
            SS = torch.sum(batch ** 2)
            # The following assum mean=0
            prdn = torch.sqrt(SE/SS).item() * 100
            snr = 10 * torch.log(SS/SE).item()
            # qs = CR / prdn
            PRDN[cnt] = prdn
            SNR[cnt] = snr
            #QS[cnt] = qs
            cnt += 1

        dual_print(f"   PRDN: {PRDN.mean():.3f}")
        dual_print(f"   SNR: {SNR.mean():.3f}")
        #dual_print(f"   QS: {QS.mean():.3f}")

        #return x_refined

    # Sample some latents and decode to EMG for sanity check
    #with torch.no_grad():
        #latents = ddpm.sample((DIFF_SAMPLES_TO_SAVE, NUM_FILTER, OUTPUT_TIME_DIM, OUTPUT_CHANNEL_DIM))  # [N,C,H,W]
        #print(latents.shape)
        """
        outs = []
        for i in range(DIFF_SAMPLES_TO_SAVE):
            vcae_model._pool_indices.clear()
            vcae_model._prepool_sizes.clear()
            out = vcae_model.decode(latents[i:i+1])  # [1,1,100,128]
            outs.append(out.cpu())
        """
        #outs = torch.cat(outs, dim=0)  # [N,1,100,128]


def check_dimension():
    model = EMG128CAE(num_pooling=NUM_POOLING, num_filter=NUM_FILTER).to(DEVICE)
    x = torch.randn(1, 1, 100, 128).to(DEVICE)
    x = x.squeeze(1).permute(0, 2, 1)
    print(f"shape: {x.shape}")
    for i, layer in enumerate(model.encoder):
        x = layer(x)
        print(f"shape of {i}: {x.shape}")
    for i, layer in enumerate(model.decoder):
        x = layer(x)
        print(f"shape of {i}: {x.shape}")
    x = x.permute(0, 2, 1).unsqueeze(1)
    print(f"shape: {x.shape}")
    exit(0)

def dual_print(mes):
    print(mes)
    with open(f"log/{NAME}.log", 'a') as f:
        f.write(str(mes) + '\n')
    
def cal_loss(x, recon, mu, logvar, beta):
    MSE = F.mse_loss(recon, x, reduction="mean")
    KL = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return MSE + beta * KL

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
                for rep in indeces:
                    idx_base = subject*dataset.subject_len + (gesture*REPETITION + rep)*dataset.window_num
                    if rep < test_bound:
                        test_idx.extend([idx_base + i for i in range(dataset.window_num)])
                    elif rep < val_bound:
                        val_idx.extend([idx_base + i for i in range(dataset.window_num)])
                    else:
                        train_idx.extend([idx_base + i for i in range(dataset.window_num)])

        #check_dimension()
        process_one_fold(train_idx, val_idx, test_idx, fold)
