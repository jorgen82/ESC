import model_predictor as model_predictor
import model_predictor_multi_class as model_predictor_multi_class

# Define the paths to the model and the input file
model_path = 'C:/Users/george.apostolakis/Downloads/ESC-50/Models/Transfer Learning/resnet_melgram_1_64x860_best_model.pth'
input_audio_path = 'C:/Users/george.apostolakis/Downloads/ESC-50/Test Sound Files/101 - Dog/1-30344-A.ogg'
#input_audio_path = 'C:/Users/george.apostolakis/Downloads/ESC-50/Test Sound Files/205 - Chirping birds/1-34495-A-Augmented.ogg'


# Run the prediction
model_predictor.main(model_path, input_audio_path)


# Run the prediction multiclass
input_audio_path_multi_class = 'C:/Users/george.apostolakis/Downloads/ESC-50/Test Sound Files/Bird-Dog.ogg'
#input_audio_path_multi_class = 'C:/Users/george.apostolakis/Downloads/ESC-50/Test Sound Files/Dog-DoorKnock.ogg'
#input_audio_path_multi_class = 'C:/Users/george.apostolakis/Downloads/ESC-50/Test Sound Files/Dog-Siren.ogg'
#input_audio_path_multi_class = 'C:/Users/george.apostolakis/Downloads/ESC-50/Test Sound Files/Dog-Chicken.ogg'
model_predictor_multi_class.main(model_path, input_audio_path_multi_class)

