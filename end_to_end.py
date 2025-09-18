import sys
import os
import random
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from sklearn.model_selection import KFold
from tqdm import tqdm
from quant_cae import EMG128CAE, INPUT_TIME_DIM, INPUT_CHANNEL_DIM
from dataset import EMG128Dataset, LatentDataset, REPETITION, SAMPLE_LEN, BIT_RESOLUTION
from plot import plot_channel, plot_heatmap, plot_metric
from utils import *
from second_diffusion_model import LatentDiffusion, Quantizer, EMA
from lstm import LSTMRefiner

# --------- Config ---------
PAPER_SETTING = False
EARLY_STOPPING = True
ALL_SUBJECT = True # inter- or intra-subject
ALL_KFOLD = True

TRAIN_AE = True
TRAIN_DIFFUSION = False # With AE frozen
TRAIN_LSTM = False
EVAL_TRAIN = True
EVAL_AE = True
EVAL_QUANT = True
EVAL_DIFFUSION = False
EVAL_LSTM = False
PRINT_TRAIN = True
PLOT_METRIC = False

KFOLDS = 18 if ALL_SUBJECT else 5 # KFold cross validation
VAL_RATIO = 0.1 # 10% training data for validation, only used for intra-subject
EPOCHS = 20 if PAPER_SETTING else 1600
EPOCHS_D = 1600
PATIENCE = 40
BATCH_SIZE = 128 if PAPER_SETTING else 10
BETA = 1
BETA_WARM_UP = 30
CRITERION = nn.MSELoss() # nn.L1Loss() nn.MSELoss(), nn.SmoothL1Loss()
LEARNING_RATE = 1e-3 if PAPER_SETTING else 2e-4
LEARNING_RATE_D = 2e-4
LAMBDA = 0.5
PRETRAIN_EPOCHS = 20

TIME_DIM = 192 # The dimension to represent the timesteps
TIME_EMB_DIM = 256 # The dimension to embed the time step for condition
NUM_POOLING = 2 # The number of pooling layer within encoder (mirrored by unpool in decoder)
NUM_FILTER = 1 # Code depth
NUM_FILTER_D = 256 # The number of filter in the unet of diffusion model
DIFFUSION_INF_STEPS = 50 # DDIM steps at inference
DIFFUSION_TRAIN_STEPS = 2500 # DDIM time steps
QUANT_BIT = 4

RANDOM_SEED = 141
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

WINDOW_SIZE = 100
SUBJECT_LIST = [x for x in range(1, 19)] if ALL_SUBJECT else [1]#[int(sys.argv[4])]
FIRST_N_GESTURE = 8
NAME = f"{sys.argv[1]}_{'all_' if ALL_SUBJECT else ''}{'paper_' if PAPER_SETTING else ''}lr{LEARNING_RATE}-{LEARNING_RATE_D}_dstep{DIFFUSION_INF_STEPS}-{DIFFUSION_TRAIN_STEPS}_e{EPOCHS}-{EPOCHS_D}_qbit{QUANT_BIT}_{NUM_POOLING}_{NUM_FILTER}"
RESULT_DIR = "cae_result"
dataset = EMG128Dataset(dataset_dir="/tmp2/b12902141/DR/CapgMyo-DB-a", window_size=WINDOW_SIZE, subject_list=SUBJECT_LIST, first_n_gesture=FIRST_N_GESTURE)
quantizer = Quantizer(num_bits=QUANT_BIT)

def dual_print(mes):
    print(mes)
    with open(f"{RESULT_DIR}/{NAME}.log", 'a') as f:
        f.write(str(mes) + '\n')
# --------------------------

def process_one_fold(train_idx, val_idx, test_idx, fold):
    fix_seed(RANDOM_SEED)
    generator = torch.Generator()
    generator.manual_seed(RANDOM_SEED)
    model = EMG128CAE(num_pooling=NUM_POOLING, num_filter=NUM_FILTER).to(DEVICE)
    if TRAIN_LSTM:
        refiner = LSTMRefiner(NUM_FILTER*64, 128, 2).to(DEVICE)
    elif TRAIN_DIFFUSION:
        latent_diffusion = LatentDiffusion(code_channels=NUM_FILTER, num_filter=NUM_FILTER_D, T=DIFFUSION_TRAIN_STEPS, time_dim=TIME_DIM, time_emb_dim=TIME_EMB_DIM).to(DEVICE)
        ema = EMA(latent_diffusion.unet, decay=0.99)


    # Training
    dual_print(f"Fold:{fold} training")
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=BATCH_SIZE, shuffle=True, generator=generator)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=BATCH_SIZE, shuffle=False, generator=generator)
    model_path = f"model2/all_fold{fold}.pth" if NUM_POOLING == 1 else f"model2/all_{NUM_POOLING}_fold{fold}.pth"
    if TRAIN_AE:
        model_state = training(model, refiner, train_loader, val_loader)
        torch.save(model_state, model_path)
        with open(f"log/{sys.argv[1]}.log", 'a') as f:
            f.write(f"Model saved for {fold} fold\n")
    else:
        model_state = torch.load(model_path)
    model.load_state_dict(model_state['model'])
    refiner.load_state_dict(model_state['refiner'])

    # Freeze model
    model.eval()
    if refiner:
        refiner.eval()

    # Evaluation
    ordered_train_loader = DataLoader(Subset(dataset, train_idx), batch_size=BATCH_SIZE, shuffle=False, generator=generator)
    test_loader = DataLoader(Subset(dataset, test_idx), batch_size=BATCH_SIZE, shuffle=False, generator=generator)
    if EVAL_TRAIN:
        evaluation(model, refiner, ordered_train_loader, name=f"{NAME}_train_{fold}")
    evaluation(model, refiner, test_loader, name=f"{NAME}_test_{fold}")

def training(model, refiner, train_loader, val_loader):
    criterion = CRITERION
    if refiner:
        optimizer = optim.Adam(list(model.parameters())+list(refiner.parameters()), lr=LEARNING_RATE)
    else:
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_loss = float('inf')
    best_model_state = None
    no_improve_epochs = 0

    for epoch in range(1, EPOCHS+1):
        # Training
        model.train()
        if refiner:
            refiner.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}, training")

        for batch in pbar:
            batch = batch.to(DEVICE) # batch: (BATCH_SIZE, 1 channel for convolution, 100, 128)
            optimizer.zero_grad()
            if model.model_type == "VCAE":
                code, mu, logvar = model.encode(batch)
                z_q = model.quant(code)
                z_ref = refiner(z_q) if refiner else z_q
                outputs = model.decode(z_ref)
                beta = min(BETA, epoch / BETA_WARM_UP * BETA)
                recon_loss = cal_loss(batch, outputs, mu, logvar, beta)
                aux_loss = criterion(z_ref, code)
            else: # CAE
                code = model.encode(batch)
                z_q = model.quant(code)
                z_ref = refiner(z_q) if refiner else z_q
                outputs = model.decode(z_ref)
                recon_loss = criterion(outputs, batch)
                aux_loss = criterion(code, z_ref)

            loss = recon_loss + LAMBDA * aux_loss
            loss.backward()
            if refiner:
                torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(refiner.parameters()), 1.0)
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(DEVICE)
                if model.model_type == "VCAE":
                    code, mu, logvar = model(batch)
                    z_q = model.quant(code)
                    z_ref = refiner(z_q) if refiner else z_q
                    outputs = model.decode(z_ref)
                    beta = min(BETA, epoch / BETA_WARM_UP * BETA)
                    loss = cal_loss(batch, outputs, mu, logvar, beta)
                else: # CAE
                    z_q = model.get_code(batch)
                    z_ref = refiner(z_q) if refiner else z_q
                    outputs = model.decode(z_ref)
                    loss = criterion(outputs, batch)
                val_loss += loss.item()

        with open(f"log/{sys.argv[1]}.log", 'a') as f:
            f.write(f"Epoch [{epoch}/{EPOCHS}], Training Loss: {train_loss/len(train_loader):.6f}, Validation Loss: {val_loss/len(val_loader):.6f}\n")

        # Early stopping
        if EARLY_STOPPING:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = {
                    'model': model.state_dict(),
                    'refiner': refiner.state_dict() if refiner else None
                }
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
        return {
            'model': model.state_dict(),
            'refiner': refiner.state_dict() if refiner else None
        }

def train_diffusion(model, latent_diffusion, ema, train_loader, val_loader, fold: int):
    global quantizer
    optimizer = optim.Adam(latent_diffusion.parameters(), lr=LEARNING_RATE_D)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=10)

    best_val_loss = float('inf')
    best_model_state = None
    no_improve_epochs = 0

    for epoch in range(1, EPOCHS_D + 1):
        latent_diffusion.train()
        running = 0.0
        pbar = tqdm(train_loader, desc=f"Diffusion Epoch {epoch}/{EPOCHS_D}")
        for code, z_q, z_clean in pbar:
            code, z_q, z_clean = code.to(DEVICE), z_q.to(DEVICE), z_clean.to(DEVICE)
            t = torch.randint(0, latent_diffusion.T, (BATCH_SIZE,), device=DEVICE, dtype=torch.long)
            if latent_diffusion.model_type == 'DECODER':
                loss = latent_diffusion.p_losses(z_clean, z_q, t)
            else:
                loss = latent_diffusion.p_losses(code, z_q, t)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(latent_diffusion.parameters(), 1.0)
            optimizer.step()
            ema.update()

            running += loss.item()
            pbar.set_postfix(loss=loss.item())
        
        # Validation
        latent_diffusion.eval()
        val_loss = 0.0
        with torch.no_grad():
            ema.copy_to_model()
            for code, z_q, z_clean in val_loader:
                code, z_q, z_clean = code.to(DEVICE), z_q.to(DEVICE), z_clean.to(DEVICE)
                t = torch.randint(0, latent_diffusion.T, (BATCH_SIZE,), device=DEVICE, dtype=torch.long)
                if latent_diffusion.model_type == 'DECODER':
                    val_loss += latent_diffusion.p_losses(z_clean, z_q, t).item()
                else:
                    val_loss += latent_diffusion.p_losses(code, z_q, t).item()
            ema.restore_model()

        scheduler.step(val_loss/len(val_loader))
        with open(f"log/{sys.argv[1]}.log", 'a') as f:
            f.write(f"[Diffusion] Epoch {epoch}, Training Loss: {running/len(train_loader):.6f}, Validation Loss: {val_loss/len(val_loader):.6f}\n")

        # Early stopping
        if EARLY_STOPPING:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = {
                    'unet': latent_diffusion.state_dict(),
                    'ema_shadow': ema.shadow_params,
                }
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
        return {
            'unet': latent_diffusion.state_dict(),
            'ema_shadow': ema.shadow_params,
        }

def train_lstm(model, lstm, train_loader, val_loader, fold):
    optimizer = optim.Adam(lstm.parameters(), lr=LEARNING_RATE_D)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=15)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    best_model_state = None
    no_improve_epochs = 0

    for epoch in range(1, EPOCHS_D + 1):
        lstm.train()
        running = 0.0
        pbar = tqdm(train_loader, desc=f"LSTM Epoch {epoch}/{EPOCHS_D}")
        for code, z_q, z_clean in pbar:
            code, z_q, z_clean = code.to(DEVICE), z_q.to(DEVICE), z_clean.to(DEVICE)
            optimizer.zero_grad()
            z_ref = lstm(z_q)
            loss = criterion(z_ref, code)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(lstm.parameters(), 1.0)
            optimizer.step()

            running += loss.item()
            pbar.set_postfix(loss=loss.item())

        # Validation
        lstm.eval()
        val_running = 0.0
        with torch.no_grad():
            for code, z_q, z_clean in val_loader:
                code, z_q, z_clean = code.to(DEVICE), z_q.to(DEVICE), z_clean.to(DEVICE)
                z_ref = lstm(z_q)
                val_loss = criterion(z_ref, code)
                val_running += val_loss.item()
        val_loss_avg = val_running / len(val_loader)
        scheduler.step(val_loss_avg)

        # Early stopping
        with open(f"log/{sys.argv[1]}.log", 'a') as f:
            f.write(f"[LSTM] Epoch {epoch} train_loss: {running/len(train_loader):.6f}, val_loss: {val_loss_avg:.6f}\n")

        if EARLY_STOPPING:
            if val_loss_avg < best_val_loss:
                best_val_loss = val_loss_avg
                best_model_state = lstm.state_dict()
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1
                if no_improve_epochs >= PATIENCE:
                    break
    if EARLY_STOPPING:
        return best_model_state
    else:
        return lstm.state_dict()

def evaluation(model, refiner, data_loader, name, plot=True):
    dual_print(f"Evaluating {name}")
    model.eval()
    if refiner:
        refiner.eval()
    def _recon(model, refiner, x): # x: (B, 1, 100, 128)
        with torch.no_grad():
            z_q = model.get_code(x)
            if refiner:
                z_ref = refiner(z_q)
            else:
                z_ref = z_q
            out = model.decode(z_ref)
        return out

    with torch.no_grad():
        batch = next(iter(data_loader)).to(DEVICE)
        if model.model_type == "VCAE":
            code, _, _ = model.encode(batch)
            code_size = code[0].numel()
        else: # CAE
            code_size = model.encode(batch)[0].numel()

        CR = INPUT_TIME_DIM * INPUT_CHANNEL_DIM / code_size * ((BIT_RESOLUTION/QUANT_BIT) if (QUANT_BIT>0) else 1)
        if plot:
            original = batch[0].squeeze().cpu().numpy()
            x = batch[0].unsqueeze(0)
            reconstructed = _recon(model, refiner, x).squeeze().cpu().numpy()
            prd = 100 * np.sqrt(np.sum((original - reconstructed) ** 2) / np.sum(original ** 2))
            dual_print(f"PRD for sample: {prd:.3f}")
            plot_channel(original, reconstructed, name)
            plot_heatmap(original, reconstructed, name)

        MSE = np.empty(len(data_loader))
        PRDN = np.empty(len(data_loader))
        SNR = np.empty(len(data_loader))
        SSIM = np.empty(len(data_loader))
        SISDR = np.empty(len(data_loader))
        QS = np.empty(len(data_loader))
        cnt = 0

        pbar = tqdm(data_loader, desc=f"Evaluation")
        for batch in pbar:
            clean = batch.to(DEVICE)  # shape: (B, 1, 100, 128)
            recon = _recon(model, refiner, clean)
            SE = torch.sum((clean-recon) ** 2)
            SS = torch.sum(clean ** 2)
            # The following assume mean=0
            mse = SE.item() / (clean.shape[0]*clean.shape[2]*clean.shape[3])
            prdn = torch.sqrt(SE/SS).item() * 100
            snr = 10 * (torch.log(SS/SE).item() if PAPER_SETTING else torch.log10(SS/SE).item())
            ssim = cal_ssim(recon, clean).mean().item()
            sisdr = cal_sisdr(recon, clean).mean().item()
            qs = CR / prdn
            MSE[cnt] = mse
            PRDN[cnt] = prdn
            SNR[cnt] = snr
            SSIM[cnt] = ssim
            SISDR[cnt] = sisdr
            QS[cnt] = qs
            cnt += 1

        dual_print(f"   CR: {CR:.2f}")
        dual_print(f"   MSE: {MSE.mean():.2f}")
        dual_print(f"   PRDN: {PRDN.mean():.3f}")
        dual_print(f"   SNR: {SNR.mean():.3f}")
        dual_print(f"   SSIM: {SSIM.mean():.3f}")
        dual_print(f"   SI-SDR: {SISDR.mean():.3f}")
        dual_print(f"   QS: {QS.mean():.3f}")
        if PLOT_METRIC:
            if len(SUBJECT_LIST) == 1:
                plot_metric(MSE.reshape(FIRST_N_GESTURE, -1), xlabel="repetition", ylabel="gesture", title=f"MSE_{name}")
                plot_metric(PRDN.reshape(FIRST_N_GESTURE, -1), xlabel="repetition", ylabel="gesture", title=f"PRDN_{name}")
                plot_metric(SNR.reshape(FIRST_N_GESTURE, -1), xlabel="repetition", ylabel="gesture", title=f"SNR_{name}")
                plot_metric(SSIM.reshape(FIRST_N_GESTURE, -1), xlabel="repetition", ylabel="gesture", title=f"SSIM_{name}")
                plot_metric(SISDR.reshape(FIRST_N_GESTURE, -1), xlabel="repetition", ylabel="gesture", title=f"SISDR_{name}")
                plot_metric(QS.reshape(FIRST_N_GESTURE, -1), xlabel="repetition", ylabel="gesture", title=f"QS_{name}")
            else:
                plot_metric(MSE.reshape(len(SUBJECT_LIST), FIRST_N_GESTURE, -1).mean(axis=2), xlabel="gesture", ylabel="subject", title=f"MSE_{name}")
                plot_metric(SNR.reshape(len(SUBJECT_LIST), FIRST_N_GESTURE, -1).mean(axis=2), xlabel="gesture", ylabel="subject", title=f"SNR_{name}")
                plot_metric(SSIM.reshape(len(SUBJECT_LIST), FIRST_N_GESTURE, -1).mean(axis=2), xlabel="gesture", ylabel="subject", title=f"SSIM{name}")
                plot_metric(SISDR.reshape(len(SUBJECT_LIST), FIRST_N_GESTURE, -1).mean(axis=2), xlabel="gesture", ylabel="subject", title=f"SISDR_{name}")
                plot_metric(QS.reshape(len(SUBJECT_LIST), FIRST_N_GESTURE, -1).mean(axis=2), xlabel="gesture", ylabel="subject", title=f"QS_{name}")

if __name__ == '__main__':
    if len(sys.argv) <= 1:
        print("Usage: python3 training.py testing_name")
        exit(1)

    Path("log").mkdir(exist_ok=True)
    Path("visual").mkdir(exist_ok=True)
    Path("model").mkdir(exist_ok=True)
    Path(RESULT_DIR).mkdir(exist_ok=True)

    for fold in range(1, KFOLDS+1):
        if not fold==10 and not ALL_KFOLD:
            continue
        with open(f"log/{sys.argv[1]}.log", 'a') as f:
            f.write(f"\nFold {fold}/{KFOLDS}\n")
        
        test_idx = []
        val_idx = []
        train_idx = []
        # non-inclusive upperbound of index of repetition for testing and validation
        test_bound = REPETITION//KFOLDS # 2
        val_bound = 2 if ALL_SUBJECT else test_bound + int(REPETITION*VAL_RATIO) # 3

        # Sample repetition for testing
        # idx: (len(SUBJECT_LIST), FIRST_N_GESTURE * REPETITION * dataset.window_num)
        rep_sample_pool = range(REPETITION) # The pool of repetition to sample from
        for subject in SUBJECT_LIST:
            for gesture in range(FIRST_N_GESTURE):
                indeces = random.sample(rep_sample_pool, REPETITION)
                for rep in indeces:
                    idx_base = subject*dataset.subject_len + (gesture*REPETITION + rep)*dataset.window_num
                    if (fold == subject and ALL_SUBJECT) or (rep < test_bound and not ALL_SUBJECT):
                        test_idx.extend([idx_base + i for i in range(dataset.window_num)])
                    elif rep < val_bound:
                        val_idx.extend([idx_base + i for i in range(dataset.window_num)])
                    else:
                        train_idx.extend([idx_base + i for i in range(dataset.window_num)])

        process_one_fold(train_idx, val_idx, test_idx, fold)
