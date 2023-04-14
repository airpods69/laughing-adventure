import numpy as np

def get_labeled_data(eeg_data, annotations):
    
    eeg_data = eeg_data.T # For easy slicing across all channels

    augmented_data = []

    for i in range(len(annotations) - 1):
        if annotations[i][0] == annotations[i + 1][0]: # same time stamp
            continue
        
        if annotations[i][-1] not in [7, 8, 9, 10]:
            continue


        low, high = annotations[i][0], annotations[i + 1][0]
        segment = (eeg_data[low: high]. T, annotations[i][-1])
        augmented_data.append(segment)

    augmented_data.append((eeg_data[annotations[-1][0]:].T, annotations[-1][-1]))
    return augmented_data
