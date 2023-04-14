import mne
import numpy as np
from get_data import get_labeled_data

from torch.utils.data import Dataset

from preprocessing import *
raw_data = mne.io.read_raw_gdf("/mnt/storage/Capstone/Projects/Datasets/BCICIV_2a_gdf/A01T.gdf")

annotations = mne.events_from_annotations(raw_data)
raw_eeg = raw_data.get_data()[:22]
labeled_eeg = get_labeled_data(raw_eeg, annotations[0])

class EEG_DATASET(Dataset):

    def __init__(self, eeg_data):
        super().__init__()
        self.eeg_data = eeg_data
        self.filtered_data = [(butter_bandpass_filter(chan[0], lowcut = 8, highcut = 30, fs = 250), chan[1]) for chan in self.eeg_data]
        self.augmented = augment_data(self.filtered_data)


    def __len__(self):
        return len(self.augmented)
    
    def __getitem__(self, idx):
        return self.augmented[idx][0], self.augmented[idx][1]

data = EEG_DATASET(labeled_eeg)
print(data.__getitem__(10))
