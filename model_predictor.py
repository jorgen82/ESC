import torch
import torch.nn.functional as F
import torchaudio
from torchaudio import transforms
import random
import warnings

warnings.filterwarnings("ignore")

#str(torchaudio.get_audio_backend())

classes = {'Dog': 18,
    'Rooster': 37,
    'Pig': 34,
    'Cow': 13,
    'Frog': 25,
    'Cat': 5,
    'Hen': 29,
    'Insects': 30,
    'Sheep': 39,
    'Crow': 16,
    'Rain': 36,
    'Sea waves': 38,
    'Crackling fire': 14,
    'Crickets': 15,
    'Chirping birds': 7,
    'Water drops': 48,
    'Wind': 49,
    'Pouring water': 35,
    'Toilet flush': 44,
    'Thunderstorm': 43,
    'Crying baby': 17,
    'Sneezing': 41,
    'Clapping': 9,
    'Breathing': 1,
    'Coughing': 12,
    'Footsteps': 24,
    'Laughing': 32,
    'Brushing teeth': 2,
    'Snoring': 42,
    'Drinking - sipping': 21,
    'Door knock': 20,
    'Mouse click': 33,
    'Keyboard typing': 31,
    'Door - wood creaks': 19,
    'Can opening': 3,
    'Washing machine': 47,
    'Vacuum cleaner': 46,
    'Clock alarm': 10,
    'Clock tick': 11,
    'Glass breaking': 26,
    'Helicopter': 28,
    'Chainsaw': 6,
    'Siren': 40,
    'Car horn': 4,
    'Engine': 22,
    'Train': 45,
    'Church bells': 8,
    'Airplane': 0,
    'Fireworks': 23,
    'Hand saw': 27}


def open(audio_path):
    sig, sr = torchaudio.load(audio_path)
    return (sig, sr)

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

def melgram(audio, n_mels=64, n_fft=1024, hop_len=256):
    sig,sr = audio
    top_db = 80
    # normalize the signal
    #normalized_sig = librosa.util.normalize(sig)
    
    # spec has shape [channel, n_mels, time], where channel is mono, stereo etc
    spec = transforms.MelSpectrogram(sr, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels)(sig)

    # Convert to decibels
    spec = transforms.AmplitudeToDB(top_db=top_db)(spec)  
    return (spec)

def normalize(melgram):
         # Calculate mean and standard deviation across all dimensions
        mean = melgram.mean()
        std = melgram.std()
        
        # Normalize the melgram
        melgram_normalized = (melgram - mean) / std
        
        return melgram_normalized
    
def transform(audio_path, model):
    try:
        aud = open(audio_path)
        reaud = resample(aud, 44100)
        rechan = rechannel(reaud, 1)
        dur_aud = pad_trunc(rechan, 5000)
        sgram = melgram(dur_aud, n_mels=64, n_fft=1024, hop_len=256)
        
        if str(model).split('(')[0] == 'ResNet':
            return sgram
        elif str(model).split('(')[0] == 'CNN':
            normalized_melgram = normalize(sgram)
            return normalized_melgram
        else: print(f"Wrong model provided...")
        raise
    except Exception as e:
        print(f"Error processing audio file {audio_path}: {e}")
        raise
    

def load_model(model_path):
    model = torch.load(model_path)
    model.eval()
    return model

def predict(audio_path, model):
    audio = transform(audio_path, model)
    audio = audio.unsqueeze(0)  # Add batch dimension

    # Perform inference
    with torch.no_grad():
        output = model(audio)
        probabilities = F.softmax(output, dim=1)
        _, predicted = torch.max(probabilities, 1)

        predicted_label = predicted.item()
        predicted_class = [key for key, x in classes.items() if x == predicted_label]
        predicted_probability = probabilities[0, predicted_label].item()
        
        # Find the index of the second highest probability
        sorted_probs, sorted_indices = torch.sort(probabilities, descending=True)
        second_predicted_label = sorted_indices[0, 1].item()
        second_predicted_class = [key for key, x in classes.items() if x == second_predicted_label]
        second_predicted_probability = sorted_probs[0, 1].item()
        
    return {
        "predicted_class": predicted_class,
        "predicted_probability": predicted_probability,
        "second_predicted_class": second_predicted_class,
        "second_predicted_probability": second_predicted_probability
    }
      

    
def main(model_path, audio_path):
    # Load the model
    model = load_model(model_path)

    # Predict the class    
    predictions = predict(audio_path, model)
    print(f"Predicted Class: {predictions['predicted_class'][0]}  with a probability of: {predictions['predicted_probability']:.04f}")
    
    if predictions['second_predicted_probability'] >= predictions['predicted_probability'] * 0.8:
        print(f"Second Predicted Class: {predictions['second_predicted_class'][0]}  with a probability of: {predictions['second_predicted_probability']:.04f}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python model_predictor.py <model_path> <audio_path>")
        sys.exit(1)
    model_path = sys.argv[1]
    audio_path = sys.argv[2]
    main(model_path, audio_path)
    
    