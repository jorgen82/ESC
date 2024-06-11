import Predict.model_predictor as model_predictor

# Define the paths to the model and the input file
model_path = 'C:/Users/george.apostolakis/Downloads/ESC-50/Models/Transfer Learning/resnet_melgram_1_64x860_best_model.pth'
input_image_path = 'C:/Users/george.apostolakis/Downloads/ESC-50/Test Sound Files/101 - Dog/1-30344-A.ogg'

# Run the prediction
model_predictor.main(model_path, input_image_path)

