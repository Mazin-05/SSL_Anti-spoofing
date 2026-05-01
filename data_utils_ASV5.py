import os
import numpy as np
import torch
import librosa
from torch.utils.data import Dataset
import warnings
from RawBoost import ISD_additive_noise, LnL_convolutive_noise, SSI_additive_noise, normWav

warnings.filterwarnings("ignore", category=UserWarning)

def genSpoof_list_ASV5(dir_meta, is_train=False, is_eval=False):
    """
    Parses the ASVspoof 5 TSV protocol files.
    Format expected: E_1607 \t E_0009538969 \t M \t C05 \t 2 \t E_0009486171 \t AC1 \t A26 \t spoof \t -
    
    Returns:
        d_meta (dict): Maps Utterance ID -> (label_integer, attack_type_string)
        file_list (list): A list of all Utterance IDs to iterate over.
    """
    d_meta = {}
    file_list = []
    
    with open(dir_meta, "r") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        cols = line.split('\t')
        
        # ASVspoof 5 full ground truth contains at least 9 columns
        if len(cols) >= 9:
            utt_id = cols[1]
            attack_type = cols[7]  # e.g., 'A26', 'A05', or 'bonafide'
            label_text = cols[8]   # 'spoof' or 'bonafide'
            
            # 1 for Bonafide (Real), 0 for Spoof (Fake)
            label = 1 if label_text.strip() == 'bonafide' else 0
            
            file_list.append(utt_id)
            d_meta[utt_id] = (label, attack_type)

    return d_meta, file_list


def pad_or_truncate(x, max_len=64600):
    """
    Ensures all audio tensors are exactly 64,600 frames long.
    This matches the specific input dimension required by the AASIST Graph Attention Network.
    """
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    
    # Pad by repeating the audio if it is too short
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x


class Dataset_ASVspoof5(Dataset):
    """
    PyTorch Dataset wrapper optimized for ASVspoof 5 directory structures,
    featuring live RawBoost data augmentation for training robustness.
    """
    def __init__(self, list_IDs, labels_dict, base_dir, args=None, algo=None, is_train=False):
        self.list_IDs = list_IDs
        self.labels_dict = labels_dict
        self.base_dir = base_dir
        self.args = args
        self.algo = algo
        self.is_train = is_train
        self.cut = 64600  # Exactly ~4 seconds of audio at 16kHz

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        utt_id = self.list_IDs[index]
        audio_path = os.path.join(self.base_dir, utt_id + ".flac")
        
        try:
            X, fs = librosa.load(audio_path, sr=16000)
            
            # Apply RawBoost Augmentation ONLY during training
            if self.is_train and self.args is not None and self.algo is not None:
                # Randomly pick an algorithm (1 to 5) as done in the original paper's robust setup
                Y = process_Rawboost_feature(X, fs, self.args, self.algo)
                X_pad = pad_or_truncate(Y, self.cut)
            else:
                X_pad = pad_or_truncate(X, self.cut)
                
        except Exception as e:
            print(f"[-] Data Corruption Warning: Could not load {audio_path}: {e}")
            X_pad = np.zeros(self.cut) # Failsafe zero-array
            
        x_inp = torch.Tensor(X_pad)
        label, attack_type = self.labels_dict[utt_id]
        
        # Return the triple-tuple for advanced metric tracking
        return x_inp, label, attack_type
    
def process_Rawboost_feature(feature, sr, args, algo):

    # Data process by Convolutive noise (1st algo)
    if algo == 1:

        feature = LnL_convolutive_noise(
            feature,
            args.N_f,
            args.nBands,
            args.minF,
            args.maxF,
            args.minBW,
            args.maxBW,
            args.minCoeff,
            args.maxCoeff,
            args.minG,
            args.maxG,
            args.minBiasLinNonLin,
            args.maxBiasLinNonLin,
            sr,
        )

    # Data process by Impulsive noise (2nd algo)
    elif algo == 2:

        feature = ISD_additive_noise(feature, args.P, args.g_sd)

    # Data process by coloured additive noise (3rd algo)
    elif algo == 3:

        feature = SSI_additive_noise(
            feature,
            args.SNRmin,
            args.SNRmax,
            args.nBands,
            args.minF,
            args.maxF,
            args.minBW,
            args.maxBW,
            args.minCoeff,
            args.maxCoeff,
            args.minG,
            args.maxG,
            sr,
        )

    # Data process by all 3 algo. together in series (1+2+3)
    elif algo == 4:

        feature = LnL_convolutive_noise(
            feature,
            args.N_f,
            args.nBands,
            args.minF,
            args.maxF,
            args.minBW,
            args.maxBW,
            args.minCoeff,
            args.maxCoeff,
            args.minG,
            args.maxG,
            args.minBiasLinNonLin,
            args.maxBiasLinNonLin,
            sr,
        )
        feature = ISD_additive_noise(feature, args.P, args.g_sd)
        feature = SSI_additive_noise(
            feature,
            args.SNRmin,
            args.SNRmax,
            args.nBands,
            args.minF,
            args.maxF,
            args.minBW,
            args.maxBW,
            args.minCoeff,
            args.maxCoeff,
            args.minG,
            args.maxG,
            sr,
        )

    # Data process by 1st two algo. together in series (1+2)
    elif algo == 5:

        feature = LnL_convolutive_noise(
            feature,
            args.N_f,
            args.nBands,
            args.minF,
            args.maxF,
            args.minBW,
            args.maxBW,
            args.minCoeff,
            args.maxCoeff,
            args.minG,
            args.maxG,
            args.minBiasLinNonLin,
            args.maxBiasLinNonLin,
            sr,
        )
        feature = ISD_additive_noise(feature, args.P, args.g_sd)

    # Data process by 1st and 3rd algo. together in series (1+3)
    elif algo == 6:

        feature = LnL_convolutive_noise(
            feature,
            args.N_f,
            args.nBands,
            args.minF,
            args.maxF,
            args.minBW,
            args.maxBW,
            args.minCoeff,
            args.maxCoeff,
            args.minG,
            args.maxG,
            args.minBiasLinNonLin,
            args.maxBiasLinNonLin,
            sr,
        )
        feature = SSI_additive_noise(
            feature,
            args.SNRmin,
            args.SNRmax,
            args.nBands,
            args.minF,
            args.maxF,
            args.minBW,
            args.maxBW,
            args.minCoeff,
            args.maxCoeff,
            args.minG,
            args.maxG,
            sr,
        )

    # Data process by 2nd and 3rd algo. together in series (2+3)
    elif algo == 7:

        feature = ISD_additive_noise(feature, args.P, args.g_sd)
        feature = SSI_additive_noise(
            feature,
            args.SNRmin,
            args.SNRmax,
            args.nBands,
            args.minF,
            args.maxF,
            args.minBW,
            args.maxBW,
            args.minCoeff,
            args.maxCoeff,
            args.minG,
            args.maxG,
            sr,
        )

    # Data process by 1st two algo. together in Parallel (1||2)
    elif algo == 8:

        feature1 = LnL_convolutive_noise(
            feature,
            args.N_f,
            args.nBands,
            args.minF,
            args.maxF,
            args.minBW,
            args.maxBW,
            args.minCoeff,
            args.maxCoeff,
            args.minG,
            args.maxG,
            args.minBiasLinNonLin,
            args.maxBiasLinNonLin,
            sr,
        )
        feature2 = ISD_additive_noise(feature, args.P, args.g_sd)

        feature_para = feature1 + feature2
        feature = normWav(feature_para, 0)  # normalized resultant waveform

    # original data without Rawboost processing
    else:

        feature = feature

    return feature