import os
import scipy.io
import torch
from torch.utils.data import Dataset

REPETITION = 10 # Repetition of each gesture
SAMPLE_LEN = 1000 # The number of sample points within a .mat file
class EMG128Dataset(Dataset):
    def __init__(self, dataset_dir, window_size=100, subject_list=list(range(1, 19)), first_n_gesture=8):
        """
        dataset_dir: The root directory containing the folder of data for each subject
        window_size: The number of samples to put in each batch
        subject_list: The list of subject (as int) to gather into the dataset
        first_n_gesture: The number of gestures to use in the dataset (at most 8)

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
        self.window_num = SAMPLE_LEN // window_size # The number of window within a .mat file (10)
        #self.window_num = 4 * SAMPLE_LEN // window_size -3 # The number of window within a .mat file (10)
        self.subject_len = first_n_gesture * REPETITION * self.window_num # The number of sample that belongs to one subject (800)
        self.samples = {i:[None for j in range(self.subject_len)] for i in subject_list} # (18, 800)
        
        for subject_folder in os.listdir(dataset_dir):
            subject_path = os.path.join(dataset_dir, subject_folder)
            if not os.path.isdir(subject_path):
                continue
            subject_index = int(subject_folder[subject_folder.index("s")+1:]) # Get the index of subject
            if not subject_index in subject_list:
                continue
            for file_name in os.listdir(subject_path):
                if not file_name.endswith('.mat'):
                    continue
                
                # Metadata
                gesture, repetition = int(file_name[4:7])-1, int(file_name[8:11])-1
                if gesture >= first_n_gesture:
                    continue
                index_base = (self.window_num * REPETITION) * gesture + self.window_num * repetition # starting index within self.samples

                # Data
                mat = scipy.io.loadmat(os.path.join(subject_path, file_name))
                data = mat['data']  # shape: (1000, 128)
                mean = data.mean(axis=0, keepdims=True)
                std = data.std(axis=0, keepdims=True)
                #mean = data.mean()
                #std = data.std()
                for i in range(0, SAMPLE_LEN - window_size + 1, window_size):
                    window = data[i:i+window_size, :] # (100, 128)
                    # channel-wise DC removal and normalization on the whole data
                    #mean = window.mean(axis=0, keepdims=True)
                    #std = window.std(axis=0, keepdims=True)
                    #mean = window.mean()
                    #std = window.std()
                    window = (window - mean) / std
                    #self.samples[subject_index][index_base + i//(window_size//4)] = torch.tensor(window, dtype=torch.float32).unsqueeze(0) # (1, 100, 128)
                    self.samples[subject_index][index_base + i//window_size] = torch.tensor(window, dtype=torch.float32).unsqueeze(0) # (1, 100, 128)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx // self.subject_len][idx % self.subject_len]

if __name__ == '__main__':
    mat = scipy.io.loadmat('CapgMyo-DB-a/dba-s1/001-001-001.mat')
    data=mat['data']
    for i in [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27]:
        print(f"{i}: {data[:8, i]*1000}")
