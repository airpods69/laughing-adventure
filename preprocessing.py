import numpy as np

from scipy.signal import butter, sosfilt, sosfreqz

def butter_bandpass(lowcut: float, highcut: float, fs: float, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        sos = butter(order, [low, high], analog=False, btype='band', output='sos')
        return sos

def butter_bandpass_filter(data: np.ndarray, lowcut: float, highcut: float, fs: float, order=5):
        sos = butter_bandpass(lowcut, highcut, fs, order=order)
        y = sosfilt(sos, data)
        return y


def split_data(data: np.ndarray):
        split = np.array_split(data.T, 5)
        return split

def augment_data(eeg_data):
        augmented = []
        
        for data in eeg_data:
            split = split_data(data[0])

            for i in split:
                augmented.append((i, data[1]))

        return augmented
