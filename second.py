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
from vcae_model import EMG128VCAE, INPUT_TIME_DIM, INPUT_CHANNEL_DIM
from dataset import EMG128Dataset, REPETITION, SAMPLE_LEN
from plot import plot_channel, plot_heatmap, plot_metric
from second_diffusion_model import LatentDiffusion, UniformQuantizer

# --------- Config ---------
PAPER_SETTING = False
EARLY_STOPPING = True
ALL_SUBJECT = False # inter- or intra-subject
ALL_KFOLD = False
PRINT_TRAIN = True
PLOT_METRIC = False

# --------- New Diffusion Config ---------
TRAIN_AE = False             # False to skip AE training if you already have weights
TRAIN_DIFFUSION = True      # Train stage-2 diffusion on latents (AE frozen)
USE_DIFFUSION_AT_TEST = False  # Route eval through diffusion sampler
DIFFUSION_STEPS = 10        # DDIM steps at inference (10–20 usually fine)
QUANT_STEP = 1.0            # Larger => stronger quantization => higher CR
QUANT_CLAMP = None          # e.g., 4.0 to clamp latent range during quantization

KFOLDS = 5 # KFold cross validation
VAL_RATIO = 0.1 # 10% training data for validation, only used for intra-subject
EPOCHS = 20 if PAPER_SETTING else 1600
PATIENCE = 40
BATCH_SIZE = 128 if PAPER_SETTING else 10
BETA = 1
BETA_WARM_UP = 30
CRITERION = nn.MSELoss() # nn.L1Loss() nn.MSELoss(), nn.SmoothL1Loss()
LEARNING_RATE = 1e-3 if PAPER_SETTING else 1e-4
WEIGHT_DECAY = 0 #1e-4

NUM_POOLING = int(sys.argv[2])
NUM_FILTER = int(sys.argv[3])

random.seed(141)
np.random.seed(141)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

WINDOW_SIZE = 100
SUBJECT_LIST = [x for x in range(1, 19)] if ALL_SUBJECT else [int(sys.argv[4])]
FIRST_N_GESTURE = 8
NAME = f"{sys.argv[1]}_e{EPOCHS}-{PATIENCE}_b{BATCH_SIZE}_lr{LEARNING_RATE}_wd{WEIGHT_DECAY}_{NUM_POOLING}_{NUM_FILTER}"
dataset = EMG128Dataset(dataset_dir="/tmp2/b12902141/DR/CapgMyo-DB-a", window_size=WINDOW_SIZE, subject_list=SUBJECT_LIST, first_n_gesture=FIRST_N_GESTURE)
quantizer = UniformQuantizer(step=QUANT_STEP, clamp=QUANT_CLAMP).to(DEVICE)
latent_diffusion = None   # lazily created after we infer code channels
# --------------------------

def process_one_fold(train_idx, val_idx, test_idx, fold):
    torch.manual_seed(fold)
    model = EMG128VCAE(num_pooling=NUM_POOLING, num_filter=NUM_FILTER).to(DEVICE)

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

        # --- Stage 2: Diffusion (latent) ---
    if USE_DIFFUSION_AT_TEST and TRAIN_DIFFUSION:
        # train using the same train_loader used for AE (we re-use variable train_loader defined earlier)
        train_diffusion(model, train_loader, fold)
    else:
        # lazy-load pretrained diffusion for this fold
        model.eval()
        tmp = next(iter(train_loader)).to(DEVICE)
        with torch.no_grad():
            z = get_code(model, tmp, deterministic_vcae=True)
        code_channels = z.shape[1]
        global latent_diffusion
        if latent_diffusion is None:
            latent_diffusion = LatentDiffusion(code_channels=code_channels).to(DEVICE)
        latent_diffusion.load_state_dict(torch.load(f"model/diffusion_latent_fold{fold}.pth", map_location=DEVICE))
        latent_diffusion.eval()

    # Evaluation
    ordered_train_loader = DataLoader(Subset(dataset, train_idx), batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(Subset(dataset, test_idx), batch_size=BATCH_SIZE, shuffle=False)
    if PRINT_TRAIN:
        evaluation(model, ordered_train_loader, name=f"{NAME}_train_{fold}")
    evaluation(model, test_loader, name=f"{NAME}_test_{fold}")
    

def training(model, train_loader, val_loader):
    criterion = CRITERION
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

def train_diffusion(model, train_loader, fold: int):
    """
    Train latent diffusion conditioned on quantized latent z_q to predict noise.
    model: AE (frozen)
    train_loader: DataLoader over training windows (same as AE training)
    """
    global latent_diffusion, quantizer
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    # infer channels from one batch
    x0 = next(iter(train_loader)).to(DEVICE)
    with torch.no_grad():
        z0 = get_code(model, x0, deterministic_vcae=True)
    code_channels = z0.shape[1]

    if latent_diffusion is None:
        latent_diffusion = LatentDiffusion(code_channels=code_channels).to(DEVICE)

    optimizer = optim.Adam(latent_diffusion.parameters(), lr=1e-4)
    EPOCHS_D = 200  # tune this

    for epoch in range(1, EPOCHS_D + 1):
        latent_diffusion.train()
        running = 0.0
        pbar = tqdm(train_loader, desc=f"Diffusion Epoch {epoch}/{EPOCHS_D}")
        for batch in pbar:
            x = batch.to(DEVICE)
            with torch.no_grad():
                z_clean = get_code(model, x, deterministic_vcae=True)  # target
                z_q = quantizer(z_clean.detach(), hard=True)          # condition (hard quant)
            B = z_clean.shape[0]
            t = torch.randint(0, latent_diffusion.T, (B,), device=DEVICE, dtype=torch.long)
            loss = latent_diffusion.p_losses(z_clean, z_q, t)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(latent_diffusion.parameters(), 1.0)
            optimizer.step()
            running += loss.item()
            pbar.set_postfix(loss=loss.item())
        dual_print(f"[Diffusion] Epoch {epoch} loss: {running/len(train_loader):.6f}")

    torch.save(latent_diffusion.state_dict(), f"model/diffusion_latent_fold{fold}.pth")

def evaluation(model, data_loader, name, plot=True):
    dual_print(f"Evaluating {name}")
    model.eval()
    def _recon_with_optional_diffusion(model, x):
        # x can be (B,1,100,128) or (1,1,100,128)
        if USE_DIFFUSION_AT_TEST and latent_diffusion is not None:
            with torch.no_grad():
                z = get_code(model, x, deterministic_vcae=True)
                z_q = quantizer(z, hard=True)
                z_hat = latent_diffusion.ddim_sample(z_q, steps=DIFFUSION_STEPS)
                y = decode_from_code(model, z_hat)
            return y
        else:
            # if VCAE forward returns (out, mu, logvar), pick out reconstruction
            if model.model_type == "VCAE":
                out, _, _ = model(x)
                return out
            else:
                return model(x)
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
            if model.model_type == "VCAE":
                #reconstructed = model(batch[0].unsqueeze(0))[0].squeeze().cpu().numpy()
                reconstructed = _recon_with_optional_diffusion(model, batch[0].unsqueeze(0)).squeeze().cpu().numpy()
            else: # CAE
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
            if model.model_type == "VCAE":
                recon = _recon_with_optional_diffusion(model, batch)
                #recon, _, _ = model(batch)
            else: # CAE
                recon = model(batch)
            SE = torch.sum((batch-recon) ** 2)
            SS = torch.sum(batch ** 2)
            # The following assum mean=0
            prdn = torch.sqrt(SE/SS).item() * 100
            snr = 10 * (torch.log(SS/SE).item() if PAPER_SETTING else torch.log10(SS/SE).item())
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

def get_code(model, x, deterministic_vcae: bool = True):
    """
    Return latent code for x. For VCAE, we use mu (deterministic) by default.
    x: tensor shape (B,1,100,128)
    """
    model.eval()
    with torch.no_grad():
        model._pool_indices.clear()
        model._prepool_sizes.clear()
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

        #check_dimension()
        process_one_fold(train_idx, val_idx, test_idx, fold)
