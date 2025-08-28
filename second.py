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
from model import EMG128CAE, INPUT_TIME_DIM, INPUT_CHANNEL_DIM
from dataset import EMG128Dataset, LatentDataset, REPETITION, SAMPLE_LEN
from plot import plot_channel, plot_heatmap, plot_metric
from utils import cal_ssim, cal_sisdr
from second_diffusion_model import LatentDiffusion, Quantizer, EMA

# --------- Config ---------
PAPER_SETTING = False
EARLY_STOPPING = True
ALL_SUBJECT = False # inter- or intra-subject
ALL_KFOLD = False

TRAIN_AE = False
TRAIN_DIFFUSION = True # With AE frozen
EVAL_TRAIN = True
EVAL_AE = True
EVAL_QUANT = True
EVAL_DIFFUSION = True
PRINT_TRAIN = True
PLOT_METRIC = False

KFOLDS = 5 # KFold cross validation
VAL_RATIO = 0.1 # 10% training data for validation, only used for intra-subject
EPOCHS = 20 if PAPER_SETTING else 1600
EPOCHS_D = 600
PATIENCE = 40
BATCH_SIZE = 128 if PAPER_SETTING else 10
BETA = 1
BETA_WARM_UP = 30
CRITERION = nn.MSELoss() # nn.L1Loss() nn.MSELoss(), nn.SmoothL1Loss()
LEARNING_RATE = 1e-3 if PAPER_SETTING else 2e-4
LEARNING_RATE_D = 2e-4
WEIGHT_DECAY = 1e-5

TIME_EMB_DIM = 128 # The dimension to represent the timesteps in cosine scheduling
NUM_POOLING = int(sys.argv[2])
NUM_FILTER = int(sys.argv[3])
NUM_FILTER_D = 128 # The number of filter in the unet of diffusion model
DIFFUSION_INF_STEPS = 50        # DDIM steps at inference
DIFFUSION_TRAIN_STEPS = 1500
QUANT_BIT = 6

random.seed(141)
np.random.seed(141)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

WINDOW_SIZE = 100
SUBJECT_LIST = [x for x in range(1, 19)] if ALL_SUBJECT else [int(sys.argv[4])]
FIRST_N_GESTURE = 8
NAME = f"{sys.argv[1]}_{'all_' if ALL_SUBJECT else ''}{'paper_' if PAPER_SETTING else ''}lr{LEARNING_RATE}-{LEARNING_RATE_D}_dstep{DIFFUSION_INF_STEPS}-{DIFFUSION_TRAIN_STEPS}_dfilter{NUM_FILTER_D}_e{EPOCHS}-{EPOCHS_D}_qbit{QUANT_BIT}_{NUM_POOLING}_{NUM_FILTER}"
RESULT_DIR = "diffusion_latent"
dataset = EMG128Dataset(dataset_dir="/tmp2/b12902141/DR/CapgMyo-DB-a", window_size=WINDOW_SIZE, subject_list=SUBJECT_LIST, first_n_gesture=FIRST_N_GESTURE)
quantizer = Quantizer(num_bits=QUANT_BIT, learn_range=False)
# --------------------------

def process_one_fold(train_idx, val_idx, test_idx, fold):
    torch.manual_seed(fold)
    model = EMG128CAE(num_pooling=NUM_POOLING, num_filter=NUM_FILTER).to(DEVICE)

    # Training
    dual_print(f"Fold:{fold} training")
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=BATCH_SIZE, shuffle=False)
    if TRAIN_AE:
        model_state = training(model, train_loader, val_loader)
        torch.save(model_state, f"model/main_fold{fold}.pth")
        with open(f"log/{sys.argv[1]}.log", 'a') as f:
            f.write(f"Model saved for {fold} fold\n")
    else:
        model_state = torch.load(f"model/main_fold{fold}.pth")
    model.load_state_dict(model_state)

    # Freeze model
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    latent_diffusion = LatentDiffusion(code_channels=NUM_FILTER, num_filter=NUM_FILTER_D, T=DIFFUSION_TRAIN_STEPS, time_dim=TIME_EMB_DIM).to(DEVICE)
    ema = EMA(latent_diffusion.unet, decay=0.99)
    latent_dataset = LatentDataset(model, dataset, DEVICE, (quantizer if (QUANT_BIT > 0) else None), SUBJECT_LIST, dataset.subject_len)
    train_loader_d = DataLoader(Subset(latent_dataset, train_idx), batch_size=BATCH_SIZE, shuffle=True)
    val_loader_d = DataLoader(Subset(latent_dataset, val_idx), batch_size=BATCH_SIZE, shuffle=False)

    if TRAIN_DIFFUSION:
        model_state = train_diffusion(model, latent_diffusion, ema, train_loader_d, val_loader_d, fold)
        torch.save(model_state, f"model/diffusion_latent_fold{fold}.pth")
    else:
        model_state = torch.load(f"model/diffusion_latent_fold{fold}.pth")
    latent_diffusion.load_state_dict(model_state['unet'])
    ema.shadow_params = [p for p in model_state['ema_shadow']]
    quantizer.load_state_dict(model_state['quantizer'])
    latent_diffusion.eval()

    # Evaluation
    ordered_train_loader = DataLoader(Subset(dataset, train_idx), batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(Subset(dataset, test_idx), batch_size=BATCH_SIZE, shuffle=False)
    if EVAL_AE:
        if EVAL_TRAIN:
            evaluation(model, latent_diffusion, ema, ordered_train_loader, quantize=False, use_diffusion=False, name=f"{NAME}_ae_train_{fold}")
        evaluation(model, latent_diffusion, ema, test_loader, quantize=False, use_diffusion=False, name=f"{NAME}_ae_test_{fold}")
    if EVAL_QUANT:
        if EVAL_TRAIN:
            evaluation(model, latent_diffusion, ema, ordered_train_loader, quantize=True, use_diffusion=False, name=f"{NAME}_quant_train_{fold}")
        evaluation(model, latent_diffusion, ema, test_loader, quantize=True, use_diffusion=False, name=f"{NAME}_quant_test_{fold}")
    if EVAL_DIFFUSION:
        if EVAL_TRAIN:
            evaluation(model, latent_diffusion, ema, ordered_train_loader, quantize=True, use_diffusion=True, name=f"{NAME}_diffusion_train_{fold}")
        evaluation(model, latent_diffusion, ema, test_loader, quantize=True, use_diffusion=True, name=f"{NAME}_diffusion_test_{fold}")

def training(model, train_loader, val_loader):
    criterion = CRITERION
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    best_val_loss = float('inf')
    best_model_state = None
    no_improve_epochs = 0

    for epoch in range(1, EPOCHS+1):
        # Training
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}, training")

        for batch in pbar:
            batch = batch.to(DEVICE) # batch: (BATCH_SIZE, 1 channel for convolution, 100, 128)
            optimizer.zero_grad()
            if model.model_type == "VCAE":
                outputs, mu, logvar = model(batch)
                beta = min(BETA, epoch / BETA_WARM_UP * BETA)
                loss = cal_loss(batch, outputs, mu, logvar, beta)
            else: # CAE
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
                if model.model_type == "VCAE":
                    outputs, mu, logvar = model(batch)
                    beta = min(BETA, epoch / BETA_WARM_UP * BETA)
                    loss = cal_loss(batch, outputs, mu, logvar, beta)
                else: # CAE
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

def train_diffusion(model, latent_diffusion, ema, train_loader, val_loader, fold: int):
    global quantizer

    optimizer = optim.Adam(latent_diffusion.parameters(), lr=LEARNING_RATE_D, weight_decay=WEIGHT_DECAY)
    #optimizer = optim.Adam(list(latent_diffusion.parameters()) + list(quantizer.parameters()), lr=LEARNING_RATE_D, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.85, patience=10)

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
                val_loss += latent_diffusion.p_losses(code, z_q, t).item()
            ema.restore_model()

        scheduler.step(val_loss/len(val_loader))
        with open(f"log/{sys.argv[1]}.log", 'a') as f:
            f.write(f"[Diffusion] Epoch {epoch}, Training Loss: {running/len(train_loader):.6f}, Validation Loss: {val_loss/len(val_loader):.6f}\n")

        # Early stopping
        if EARLY_STOPPING:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                #best_model_state = diffusion.state_dict()
                best_model_state = {
                    'unet': latent_diffusion.state_dict(),
                    'ema_shadow': ema.shadow_params,
                    'quantizer': quantizer.state_dict()
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
            'quantizer': quantizer.state_dict()
        }

def evaluation(model, latent_diffusion, ema, data_loader, name, quantize: bool, use_diffusion: bool, plot=True):
    dual_print(f"Evaluating {name}")
    model.eval()
    def _recon_with_selected_model(model, x): # x: (?, 1, 100, 128)
        with torch.no_grad():
            if not quantize:
                if model.model_type == "VCAE":
                    return model(x)[0]
                else: #CAE
                    return model(x)
            elif use_diffusion:
                latent_diffusion.eval()
                z_clean = get_code(model, x)
                if QUANT_BIT > 0:
                    z_q = quantizer(z_clean)
                ema.copy_to_model()
                z_hat = latent_diffusion.ddim_sample(z_q, steps=DIFFUSION_INF_STEPS)
                ema.restore_model()
                return decode_from_code(model, z_hat)
            else:
                z = get_code(model, x, deterministic_vcae=True)
                z_q = quantizer(z)
                return decode_from_code(model, z_q)

    with torch.no_grad():
        batch = next(iter(data_loader)).to(DEVICE)
        if model.model_type == "VCAE":
            # Be careful when calling the encoder and decoder individually, which may cause memory leakage or incorrect result
            code, mu, logvar = model.encode(batch)
            code_size = code[0].numel()
        else: # CAE
            code_size = model.encode(batch)[0].numel()

        CR = INPUT_TIME_DIM * INPUT_CHANNEL_DIM / code_size # Not considering quantization
        #CR = batch.shape[0] / model.encoder(batch).numel()
        #CR = batch.shape[0] / model.encoder(batch.squeeze(1).permute(0, 2, 1)).numel()
        if plot:
            original = batch[0].squeeze().cpu().numpy()
            reconstructed = _recon_with_selected_model(model, batch[0].unsqueeze(0)).squeeze().cpu().numpy()
            prd = 100 * np.sqrt(np.sum((original - reconstructed) ** 2) / np.sum(original ** 2))
            dual_print(f"PRD for sample: {prd:.3f}")
            plot_channel(original, reconstructed, name)
            plot_heatmap(original, reconstructed, name)

        PRDN = np.empty(len(data_loader))
        SNR = np.empty(len(data_loader))
        SSIM = np.empty(len(data_loader))
        SISDR = np.empty(len(data_loader))
        QS = np.empty(len(data_loader))
        cnt = 0
        for batch in data_loader:
            batch = batch.to(DEVICE)  # shape: (dataset.window_num, 1, 100, 128)
            recon = _recon_with_selected_model(model, batch)
            SE = torch.sum((batch-recon) ** 2)
            SS = torch.sum(batch ** 2)
            # The following assum mean=0
            prdn = torch.sqrt(SE/SS).item() * 100
            snr = 10 * (torch.log(SS/SE).item() if PAPER_SETTING else torch.log10(SS/SE).item())
            ssim = cal_ssim(recon, batch).mean().item()
            sisdr = cal_sisdr(recon, batch).mean().item()
            # qs = CR / prdn
            PRDN[cnt] = prdn
            SNR[cnt] = snr
            SSIM[cnt] = ssim
            SISDR[cnt] = sisdr
            #QS[cnt] = qs
            cnt += 1

        dual_print(f"   CR: {CR:.2f}")
        dual_print(f"   PRDN: {PRDN.mean():.3f}")
        dual_print(f"   SNR: {SNR.mean():.3f}")
        dual_print(f"   SSIM: {SSIM.mean():.3f}")
        dual_print(f"   SI-SDR: {SISDR.mean():.3f}")
        #dual_print(f"   QS: {QS.mean():.3f}")
        if PLOT_METRIC:
            if len(SUBJECT_LIST) == 1:
                plot_metric(PRDN.reshape(FIRST_N_GESTURE, -1), xlabel="repetition", ylabel="gesture", title=f"PRDN_{name}")
                plot_metric(SNR.reshape(FIRST_N_GESTURE, -1), xlabel="repetition", ylabel="gesture", title=f"SNR_{name}")
                plot_metric(SSIM.reshape(FIRST_N_GESTURE, -1), xlabel="repetition", ylabel="gesture", title=f"SSIM_{name}")
                plot_metric(SISDR.reshape(FIRST_N_GESTURE, -1), xlabel="repetition", ylabel="gesture", title=f"SISDR_{name}")
                #plot_metric(QS.reshape(FIRST_N_GESTURE, -1), xlabel="repetition", ylabel="gesture", title=f"QS_{name}")
            else:
                plot_metric(PRDN.reshape(len(SUBJECT_LIST), FIRST_N_GESTURE, -1).mean(axis=2), xlabel="gesture", ylabel="subject", title=f"PRDN_{name}")
                plot_metric(SNR.reshape(len(SUBJECT_LIST), FIRST_N_GESTURE, -1).mean(axis=2), xlabel="gesture", ylabel="subject", title=f"SNR_{name}")
                plot_metric(SSIM.reshape(len(SUBJECT_LIST), FIRST_N_GESTURE, -1).mean(axis=2), xlabel="gesture", ylabel="subject", title=f"SSIM{name}")
                plot_metric(SISDR.reshape(len(SUBJECT_LIST), FIRST_N_GESTURE, -1).mean(axis=2), xlabel="gesture", ylabel="subject", title=f"SISDR_{name}")
                #plot_metric(QS.reshape(len(SUBJECT_LIST), FIRST_N_GESTURE, -1).mean(axis=2), xlabel="gesture", ylabel="subject", title=f"QS_{name}")

def dual_print(mes):
    print(mes)
    with open(f"{RESULT_DIR}/{NAME}.log", 'a') as f:
        f.write(str(mes) + '\n')
    
def cal_loss(x, recon, mu, logvar, beta):
    MSE = F.mse_loss(recon, x, reduction="mean")
    KL = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return MSE + beta * KL

def get_code(model, x, deterministic_vcae: bool = True):
    """
    Return latent code for x. For VCAE, we use mu (deterministic) by default.
    x: tensor shape (B,1,100,128)
    """
    model.eval()
    with torch.no_grad():
        if model.model_type == "VCAE":
            code, mu, logvar = model.encode(x)
            return mu if deterministic_vcae else code
        else:
            return model.encode(x)

def decode_from_code(model, z):
    model.eval()
    with torch.no_grad():
        return model.decode(z)

if __name__ == '__main__':
    if len(sys.argv) <= 1:
        print("Usage: python3 training.py testing_name")
        exit(1)

    for fold in range(1, KFOLDS+1):
        if fold==3 and not ALL_KFOLD:
            break
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
