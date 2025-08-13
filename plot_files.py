# plot_mat.py
import sys
import os
import scipy.io
import numpy as np
import matplotlib.pyplot as plt

# ==== USER-DEFINED GLOBAL COLOR SCALE ====
VMIN = -0.5  # e.g., 0
VMAX = 0.5  # e.g., 3000
# =========================================

def plot_mat_file(file_path):
    if not os.path.isfile(file_path):
        print(f"[Warning] {file_path} is not a valid file.")
        return
    
    mat = scipy.io.loadmat(file_path)
    if "data" not in mat:
        print(f"[Warning] No 'data' key found in {file_path}. Available keys: {mat.keys()}")
        return
    
    data = mat["data"]  # Expected shape: (1000, 128)
    if data.ndim != 2:
        print(f"[Warning] Unexpected shape {data.shape} in {file_path}")
        return
    
    plt.figure(figsize=(10, 6))
    plt.imshow(data.transpose(), aspect='auto', cmap='viridis', origin='lower',
               vmin=VMIN, vmax=VMAX)
    plt.colorbar(label='Amplitude')
    plt.ylabel("Channel (1–128)")
    plt.xlabel("Time sample (1–1000)")
    plt.title(os.path.basename(file_path))
    plt.tight_layout()
    #plt.show()
    plt.savefig(f"/tmp2/b12902141/DR/visual/file_{file_path}.png")
    plt.close()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_mat.py file1.mat file2.mat ...")
        sys.exit(1)
    
    for file_path in sys.argv[1:]:
        plot_mat_file(file_path)

