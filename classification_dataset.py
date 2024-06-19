"""
    Its used to create the dataset, by applying all the selected preprocessing steps.

Returns:
    spectogram and label
"""

from torch.utils.data import DataLoader, Dataset, random_split
import torch.nn
#import torchaudio
from processing import AudioUtil

# ----------------------------
# Sound Dataset
# ----------------------------
class SoundCustomDS(Dataset):
    def __init__(self, df, duration, sr, channel):
        self.df = df
        self.duration = duration
        self.sr = sr
        self.channel = channel
        #self.shift_pct = 0.4
                
    # ----------------------------
    # Number of items in dataset
    # ----------------------------
    def __len__(self):
        return len(self.df)    
        
    # ----------------------------
    # Get i'th item in dataset
    # ----------------------------
    def __getitem__(self, idx):
        audio_file = self.df.loc[idx, 'Subfolder_Path'] + '/' + self.df.loc[idx, 'File Name']
        label = self.df.loc[idx, 'label']
        audio_util = AudioUtil()
        aud = audio_util.open(audio_file)
        reaud = audio_util.resample(aud, self.sr)
        rechan = audio_util.rechannel(reaud, self.channel)
        dur_aud = audio_util.pad_trunc(rechan, self.duration)
        sgram = audio_util.melgram(dur_aud, n_mels=64, n_fft=1024, hop_len=256)
        num_channels, num_mels, num_frames = sgram.shape
        
        # Interpolate dataset (from (64, 860) to (860, 860)
        #mel_spectrogram_resized = torch.nn.functional.interpolate(sgram.unsqueeze(0), size=(num_frames, num_frames), mode='bicubic').squeeze(0)
        
        # Mask the sgram
        #aug_sgram = audio_util.melgram_augment(sgram, max_mask_pct=0.1, n_freq_masks=2, n_time_masks=2)
        
        normalized_melgram = audio_util.normalize(sgram)
        return normalized_melgram, label

