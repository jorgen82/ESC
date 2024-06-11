There are 4 basic files
1) main.py : This is to train/load/test the custom CNN
2) resnet18.py : This is for the pretrained resnet18 model
3) resnet and cnn.py : This is for the combination of resnet18 with a custom cnn
4) SVM.py : This is for a simple svm using mfcc

Other than that we have
1) classification_dataset.py : This creates the dataset for the training. It will do all the necessary transforamtions.
2) classification_model.py : All the classification models used are defined here
3) initialize.py : This was used as the initialization step of our experiment. It will export the melgrams/waves/mfcc visuals for inspection, do the data augmentation and create the test dataset.
4) plot_results.py : Ploting of the train/val accuracy and loss charts along with the confusion matrix.
5) processing.py : All the data processing (data load, rechannel, resample, pad-trancate, export melgram, add noise, time shift, time stretching.

Also under predict there is a main file called run_prediction.py which will load the model and test file, based on their input paths and return the prediction.

Finally, due to size limitation, we only uploaded the custom cnn model.
