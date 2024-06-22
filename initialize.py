"""
    This is used for the initial steps of the project.
    1) FileMover: Moves the test files to separate directory based on the subset_percentage (train/test split).
    2) initialize_folder: Deletes the directory contents and creates the structure. used for exporting the images as per next functions.
    3) create_images: Create and export visuals for wave, melgram and MFCC.
    4) file_exploration: Records all the necessary details of the files and calls the create_images in order to export the visuals.
    5) combine_audio_visuals: Create a big visual which contains the visuals for all the files. This is to be able to have a first look on possible similarities.
    6) data_augmentation: Create new files by apply some processing steps

"""

import os
import shutil
import random
import pandas as pd
from pathlib import Path
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import sys
from processing import AudioUtil
import torchaudio
import soundfile as sf

# Moves the test files to separate directory based on the subset_percentage (train/test split).
# source_dir: The source directory
# destination_dir: The directory to move the files to
# subset_percentage: The percentage of the files to be moved
# augment_data: If we will augment data (therefore the remaining date will be duplicated), then we will mupliply the subset_percentage with 1.625 in order to able to get a move percentage close to the one we want
class FileMover:
    def __init__(self, df, source_dir, destination_dir, subset_percentage, augment_data):
        self.df = df
        self.source_dir = source_dir
        self.destination_dir = destination_dir
        self.augment_pct = 1.625 if augment_data else 1
        self.subset_percentage = subset_percentage * self.augment_pct

    def move_files_with_subset(self):
        if os.path.exists(self.destination_dir):
            shutil.rmtree(self.destination_dir)

        os.makedirs(self.destination_dir)

        for root, dirs, files in os.walk(self.source_dir):
            dest_root = root.replace(self.source_dir, self.destination_dir)
            os.makedirs(dest_root, exist_ok=True)

        for i in self.df['Subfolder'].unique():
            sampled_rows = self.df[self.df['Subfolder'] == i].sample(frac=self.subset_percentage)
            sampled_rows['train_test'] = 'test'
            self.df.update(sampled_rows)
            
        for index, row in self.df[self.df['train_test'] == 'test'].iterrows():
            dest_path = self.destination_dir + '/' + row['Subfolder']
            file = row['File Name']
            src_file = os.path.join(row['Subfolder_Path'], file)
            dest_file = os.path.join(dest_path, file)
            shutil.move(src_file, dest_file)
            self.df.at[index, 'Subfolder_Path'] = row['Subfolder_Path'].replace(self.source_dir, self.destination_dir)


# Deletes the directory contents and creates the structure. used for exporting the images as per next functions.               
def initialize_folder(output_root_folder):
    if os.path.exists(output_root_folder):
        for root, dirs, files in os.walk(output_root_folder):
            for f in files:
                try:
                    os.unlink(os.path.join(root, f))
                except Exception as e:
                    print(f'Failed to delete file: {e}')
                    sys.exit(0)   
            for d in dirs:
                try:
                    shutil.rmtree(os.path.join(root, d))
                except Exception as e:
                    print(f'Failed to delete directory: {e}') 
                    sys.exit(0)

# Create and export visuals for wave, melgram and MFCC.
def create_images(audio_file, output_path):
    y, sr = librosa.load(audio_file)  # Load the audio file
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)  # Compute mel spectrogram
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)  # Convert to log scale
    mfccs = librosa.feature.mfcc(y=y, sr=sr)  # Compute the MFCC
    
    # Save the mel spectrogram
    plt.figure(figsize=(8, 4))
    librosa.display.specshow(log_mel_spec, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel Spectrogram')
    plt.savefig(output_path[0])  # Save as image file
    plt.close()
    
    # Save the wave
    plt.figure(figsize=(8, 4))
    librosa.display.waveshow(y, sr=sr)
    plt.title('Amplitude vs Time')
    plt.savefig(output_path[1])  # Save as image file
    plt.close()

    # Save the MFCC
    plt.figure(figsize=(8, 4))
    librosa.display.specshow(mfccs, x_axis='time', sr=sr)
    plt.colorbar(format='%+2.0f dB')
    plt.title('MFCC')
    plt.xlabel('Time')
    plt.ylabel('MFCC Coefficients')
    plt.savefig(output_path[2])  # Save as image file
    plt.close()
    
    # Save the combination
    # Create a figure with two subplots
    fig, axes = plt.subplots(3, 1, figsize=(3, 3))
    # Plot the spectrogram on the first axis
    librosa.display.specshow(log_mel_spec, sr=sr, x_axis=None, y_axis=None, ax=axes[0])
    # Plot the waveform on the second axis
    librosa.display.waveshow(y, sr=sr, ax=axes[1])
    # Plot the mfcc on the third axis
    librosa.display.specshow(mfccs, sr=sr, ax=axes[2])
    # Remove axis labels and values
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    axes[1].set_xlabel('')
    # Set titles
    axes[0].set_title(os.path.basename(audio_file))
    # Remove spacing between subplots
    plt.subplots_adjust(hspace=0)
    plt.savefig(output_path[3])  # Save as image file
    plt.close()
    

# Records all the necessary details of the files and calls the create_images in order to export the visuals.
# source_dir: The directory to scan and get the file info.
# output_root_folders: The list of folders that the audio visuals will be exported.
# file_name_filter: If we look for files with specific name. This is used to be called for gathering the info for the Augmented files, since augmentation will be after the initial gathering of information and the test file move takes place.
# export_audio_visuals: If we want to export the audio visuals, or jsut to create the dataframe with the info
def file_exploration(source_dir, output_root_folders, file_name_filter = None, export_audio_visuals = True):
    audio_extensions = ['.mp3', '.wav', '.flac', '.ogg']  # Add more extensions if needed
    matrix = []
    # If we want to export audio visuals, we first delete output folder contents
    if export_audio_visuals:
        for folder in output_root_folders: initialize_folder(folder)
        for folder in output_root_folders: os.makedirs(folder, exist_ok=True)
        
    for root, dirs, _ in os.walk(source_dir):
        for subfolder in dirs:
            subfolder_path = os.path.join(source_dir, subfolder).replace("\\", "/")
            for file in os.listdir(subfolder_path):
                if os.path.isfile(os.path.join(subfolder_path, file)) and os.path.splitext(file)[1] in audio_extensions:
                    if file_name_filter is None or file_name_filter in file:
                        audio_file = os.path.join(subfolder_path, file).replace("\\", "/")
                        audio_data, sample_rate = librosa.load(os.path.join(audio_file))
                        duration = librosa.get_duration(y=audio_data, sr=sample_rate)
                        channels = len(audio_data.shape)
                        matrix.append([subfolder, subfolder_path, file, os.path.splitext(file)[1][1:], duration, channels, sample_rate])
        
                        if export_audio_visuals:
                            image_files = []  
                            image_folders = []                     
                            for folder in output_root_folders: 
                                image_folder = folder + '/' + subfolder
                                image_folders.append(image_folder) 
                                os.makedirs(image_folder, exist_ok=True) 
                                file_name = os.path.splitext(file)[0]
                                image_files.append(image_folder + '/' + f"{file_name}.jpg")
                            create_images(audio_file, image_files)
    return matrix


#Create a big visual which contains the visuals for all the files. This is to be able to have a first look on possible similarities.
def combine_audio_visuals(folder, output_folder):
    # Get all subfolders in the root folder
    subfolders = [f.path for f in os.scandir(folder) if f.is_dir()]
    max_files = max([len(files) for r, d, files in os.walk(folder)])
        
    # Calculate the total number of subplots
    total_subplots = len(subfolders) * max_files
    
    # Create a figure with a fixed width
    fig_width = 4 * max_files  # Adjust the figure width as needed
    fig, axes = plt.subplots(len(subfolders), max_files, figsize=(fig_width, 4.5*len(subfolders)))
    
    # Adjust the horizontal spacing between subplots
    fig.subplots_adjust(wspace=0.1)
    
    # Plot images from each subfolder
    for i, subfolder in enumerate(subfolders):
        # Get all image files in the subfolder
        image_files = [f.path for f in os.scandir(subfolder) if f.is_file() and f.name.endswith('.jpg')]
        
        # Plot each image in the subfolder
        for j, image_file in enumerate(image_files):
            image = plt.imread(image_file)
            axes[i, j].imshow(image)
            axes[i, j].axis('off')
            
        # Set the title of the subplot as the subfolder name aligned to the left
        axes[i, 0].set_title(os.path.basename(subfolder), loc='left')
    plt.savefig(output_folder + "ComboPlot.jpg")
    plt.show()
    

# Create new files by apply some processing steps
def data_augmentation(source_dir):
    audio_extensions = ['.mp3', '.wav', '.flac', '.ogg']  # Add more extensions if needed
    for root, dirs, _ in os.walk(source_dir):
        for subfolder in dirs:
            subfolder_path = os.path.join(source_dir, subfolder).replace("\\", "/")
            for file in os.listdir(subfolder_path):
                if os.path.isfile(os.path.join(subfolder_path, file)) and os.path.splitext(file)[1] in audio_extensions:
                    audio_util = AudioUtil()
                    audio_file = os.path.join(subfolder_path, file).replace("\\", "/")
                    audio, sr = torchaudio.load(audio_file)
                    random_shift = audio_util.time_shift((audio, sr), 0.4)
                    rechan = audio_util.rechannel(random_shift, 1)
                    noise_data = audio_util.add_noise(rechan)
                    volume_scaling = audio_util.volume_scaling(noise_data)
                    time_stretching = audio_util.time_stretching(volume_scaling, rate=1.5)
                    #pad_trunc = audio_util.pad_trunc(time_stretching, 5000)
                    return_audio = time_stretching[0].numpy()
                    return_sr = time_stretching[1]
                    new_filename = audio_file[:-4] + '-Augmented' +audio_file[-4:]
                    if audio_file[-3:] == 'ogg':
                        sf.write(new_filename, return_audio[0], return_sr, format='ogg', subtype='vorbis')
                    else:
                        sf.write(new_filename, return_audio[0], return_sr, format=audio_file[-3:])
