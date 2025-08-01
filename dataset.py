import os
import scipy.io
import torch
from torch.utils.data import Dataset

class EMG128Dataset(Dataset):
    def __init__(self, dataset_dir, window_size=100, subject_list=list(range(1, 19))):
        """
        dataset_dir: The root directory containing the folder of data for each subject
        window_size: The number of samples to put in each batch
        subject_list: The list of subject (as int) to gather into the dataset

        File system looks like:
        dataset_dir (CapgMyo-DB-a)
        ├── dba-s1
        │   ├── 001-001-001.mat
        │   ├── 001-001-002.mat
        ......
        │   └── 001-008-010.mat 
        ├── dba-s2
        │   ├── 002-001-001.mat
        ......
        """
        self.samples = []
        
        for subject_folder in os.listdir(dataset_dir):
            subject_path = os.path.join(dataset_dir, subject_folder)
            if not os.path.isdir(subject_path):
                continue
            subject_index = int(subject_folder[subject_folder.index("s")+1:]) # Get the index of subject
            if not subject_index in subject_list:
                continue
            for file_name in os.listdir(subject_path):
                if file_name.endswith('.mat'):
                    mat = scipy.io.loadmat(os.path.join(subject_path, file_name))
                    data = mat['data']  # shape: (1000, 128)
                    for i in range(0, 1000 - window_size + 1, window_size):
                        metadata = [subject_index, int(file_name[4:7]), int(file_name[8:10])] # [subject, gesture, repetition]
                        window = data[i:i+window_size, :] # (100, 128)
                        # DC removal (channel-wise normalization)
                        # window = (window - window.mean(axis=0, keepdims=True)) / window.std(axis=0, keepdims=True)
                        window -= window.mean(axis=0, keepdims=True)
                        self.samples.append([torch.tensor(window, dtype=torch.float32).unsqueeze(0), metadata]) # (1, 100, 128)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
