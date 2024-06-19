"""
    Contains the preprocessing steps and the Augmentation techniques
    Preprocessing:
        1) AudioUtil: Load the audio file
        2) rechannel: Convert the audio to the desired number of channels
        3) resample: Standardise sampling rate
        4) pad_trunc: Pad or truncate to a 'max_ms' in milliseconds
        5) melgram: Generate a Melgram
        6) normalize: Normalize the output
    Augmentation:
        1)time_shift: Shifts the signal to the left or right by some percent and wrap around if needed
        2) add_noise: Add noise to the audio signal to simulate different noise conditions.
        3) volume_scaling: Change the volume of the audio signal by scaling it up or down. 
        4) time_stretching: Change the duration of the audio signal by speeding it up or slowing it down. 
        5) melgram_augment: Augment the Spectrogram by masking out some sections of it in both the frequency dimension
"""

import random
import numpy as np
import librosa
import torch
import torchaudio
from torchaudio import transforms
from IPython.display import Audio
import torch.nn.functional as F

# Load and audio file and return the signal as a tensor, along with the sample rate
class AudioUtil():
    @staticmethod
    def open(audio_file):
        sig, sr = torchaudio.load(audio_file)
        return (sig, sr)


    # Convert the audio to the desired number of channels
    @staticmethod
    def rechannel(audio, new_channel):
        sig, sr = audio

        if (sig.shape[0] == new_channel):
            return audio

        if (new_channel == 1):
            # Convert from stereo to mono by selecting only the first channel
            resig = sig[:1, :]
        else:
            # Convert from mono to stereo by duplicating the first channel
            resig = torch.cat([sig, sig])

        return ((resig, sr))

        
    # Standardise sampling rate
    # Since Resample applies to a single channel, we resample one channel at a time
    @staticmethod
    def resample(audio, new_sr):
        sig, sr = audio

        if (sr == new_sr):
            return audio

        num_channels = sig.shape[0]
        # Resample first channel
        resig = torchaudio.transforms.Resample(sr, new_sr)(sig[:1,:])
        if (num_channels > 1):
            # Resample the second channel and merge both channels
            retwo = torchaudio.transforms.Resample(sr, new_sr)(sig[1:,:])
            resig = torch.cat([resig, retwo])

        return ((resig, new_sr))
    
    
    # Pad or truncate to a 'max_ms' in milliseconds
    @staticmethod
    def pad_trunc(audio, max_ms):
        sig, sr = audio
        num_rows, sig_len = sig.shape
        max_len = sr//1000 * max_ms

        if (sig_len > max_len):
            # Truncate the signal to the given length
            sig = sig[:,:max_len]

        elif (sig_len < max_len):
            # Length of padding to add at the beginning and end of the signal
            pad_begin_len = random.randint(0, max_len - sig_len)
            pad_end_len = max_len - sig_len - pad_begin_len

            # Pad with 0s
            pad_begin = torch.zeros((num_rows, pad_begin_len))
            pad_end = torch.zeros((num_rows, pad_end_len))

            sig = torch.cat((pad_begin, sig, pad_end), 1)
        return (sig, sr)

    # Generate a Melgram
    @staticmethod
    def melgram(audio, n_mels=64, n_fft=1024, hop_len=256):
        sig,sr = audio
        top_db = 80

        # normalize the signal
        #normalized_sig = librosa.util.normalize(sig)
        
        # spec has shape [channel, n_mels, time], where channel is mono, stereo etc
        spec = transforms.MelSpectrogram(sr, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels)(sig)

        # Convert to decibels
        spec = transforms.AmplitudeToDB(top_db=top_db)(spec)
        #spec = transforms.AmplitudeToDB()(spec)        
        return (spec)
    
    # Normalize the output
    @staticmethod
    def normalize(melgram):
         # Calculate mean and standard deviation across all dimensions
        mean = melgram.mean()
        std = melgram.std()
        
        # Normalize the melgram
        melgram_normalized = (melgram - mean) / std
        
        return melgram_normalized
    
       
    ########################
    # Data Augmentation
    ########################
    # Shifts the signal to the left or right by some percent and wrap around if needed
    @staticmethod
    def time_shift(audio, shift_limit):
        sig,sr = audio
        _, sig_len = sig.shape
        shift_amt = int(random.random() * shift_limit * sig_len)
        return torch.roll(sig, shifts=shift_amt, dims=1), sr
    
    # Add noise : This technique adds noise to the audio signal to simulate different noise conditions.
    @staticmethod
    def add_noise(audio):
        sig, sr = audio
        #noise = np.random.normal(0, 0.1, len(sig))
        noise = torch.tensor(np.random.normal(0, 0.1, len(sig)), dtype=sig.dtype)
        audio_noisy = sig + noise
        return audio_noisy, sr

    # Volume scaling: This technique changes the volume of the audio signal by scaling it up or down. 
    # This can be useful for simulating variations in the loudness of the audio.
    @staticmethod
    def volume_scaling(audio):
        sig, sr = audio
        dyn_change = torch.tensor(np.random.uniform(low=1.5,high=2.5))
        data = sig * dyn_change
        return data, sr
    
    # Time stretching : This technique changes the duration of the audio signal by speeding it up or slowing it down. 
    # This can be useful for simulating variations in the tempo of the audio.
    @staticmethod
    def time_stretching(audio, rate):
        sig, sr = audio
        # Compute the new length of the audio
        new_length = int(sig.size(-1) * rate)
        # Apply linear interpolation to stretch the audio
        stretched_audio = F.interpolate(sig.unsqueeze(0), size=new_length, mode='linear').squeeze(0)
        return stretched_audio, sr
    
    # Augment the Spectrogram by masking out some sections of it in both the frequency dimension (ie. horizontal bars) 
    # and the time dimension (vertical bars) to prevent overfitting 
    @staticmethod
    def melgram_augment(spec, max_mask_pct=0.1, n_freq_masks=1, n_time_masks=1):
        _, n_mels, n_steps = spec.shape
        mask_value = spec.mean()
        aug_spec = spec

        freq_mask_param = max_mask_pct * n_mels
        for _ in range(n_freq_masks):
            aug_spec = transforms.FrequencyMasking(freq_mask_param)(aug_spec, mask_value)

        time_mask_param = max_mask_pct * n_steps
        for _ in range(n_time_masks):
            aug_spec = transforms.TimeMasking(time_mask_param)(aug_spec, mask_value)
        return aug_spec



    
     