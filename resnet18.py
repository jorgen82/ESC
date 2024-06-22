"""
    Basically this is like the main script, but it utilizes the pretrained resnet18 model
    1) Load or create the dataframe with the file info (based on the read_dataset)
        In case of creation, it will create the dataframe and export the visuals (if export_audio_visuals = True) 
        It will also create augmented files (if augment_data = True)
    2) We will see some simple visuals and info in order to see the class distibution along with duration and channel info.
    3) We define the train and validate functions.
        Training and validation will be handled by train_val function
        train_val is called by the objective function, which is the one which will handle the Optuna trials
    4) Training results will be displayed along with the best hyperparameters, validation accuracy and loss visuals
        Hyperparameters and results of each epoch (acc, loss) will also be saved to files for later evaluation of our different experiments
    5) Test the model
        Also show the confusion matrix
        Create a dataframe with all the probalities and the predicted class (also shows ore than one class if closer to a threshold with the predicted)
     
"""
import pandas as pd
from initialize import file_exploration, combine_audio_visuals, data_augmentation, FileMover
from sklearn import preprocessing 

# Create or Read dataset
read_dataset = True
export_audio_visuals = True
augment_data = False

# Path initialization
model_name = 'pretrained_resnet18'
root_path = 'C:/Users/george.apostolakis/Downloads/ESC-50/'
source_directory = root_path + 'Sound Files'
melgram_output_root_folder = root_path + 'Melgrams'
wave_output_root_folder = root_path + 'Waves'
mfcc_output_root_folder = root_path + 'MFCC'
comboplots_output_root_folder = root_path + 'ComboPlots'
dataframe_csv = root_path + 'dataframe.csv'
test_files_directory = root_path + 'Test Sound Files'
models_directory = root_path + 'Models/'
output_root_folders = [melgram_output_root_folder,wave_output_root_folder,mfcc_output_root_folder,comboplots_output_root_folder]

if read_dataset:
    # Read dataframe
    df = pd.read_csv(dataframe_csv)
else:
    # Fetch audio file info and export the visuals (if export_audio_visuals = True)
    df = pd.DataFrame(file_exploration(source_dir = source_directory, output_root_folders = output_root_folders, file_name_filter=None, export_audio_visuals = export_audio_visuals), columns=['Subfolder', 'Subfolder_Path', 'File Name', 'Extension', 'Duration', 'Channels', 'SampleRate'])
    
    # Move test data to the test_files_directory
    subset_percentage = 0.2
    df['train_test'] = 'train'
    file_mover = FileMover(df, source_directory, test_files_directory, subset_percentage, augment_data)
    file_mover.move_files_with_subset()
    
    # Augment audio files
    if augment_data: data_augmentation(source_directory) 
    
    # Read the info of the Augmented files
    df_augmented = pd.DataFrame(file_exploration(source_dir = source_directory, output_root_folders = output_root_folders, file_name_filter='Augmented', export_audio_visuals = export_audio_visuals), columns=['Subfolder', 'Subfolder_Path', 'File Name', 'Extension', 'Duration', 'Channels', 'SampleRate'])
    df_augmented['train_test'] = 'train'
    
    # Union the datasets
    result = pd.concat([df, df_augmented], ignore_index=True)
    df = result.copy()
    
    # Create class and label column (label will be created by encoding the class to be used in training)
    df['class'] = df['Subfolder'].astype(str).str[6:]
    label_encoder = preprocessing.LabelEncoder() 
    df['label']= label_encoder.fit_transform(df['class']) 
    
    # Combine visuals
    if export_audio_visuals: combine_audio_visuals(comboplots_output_root_folder, root_path)
    
    # Save dataframe
    df.to_csv(dataframe_csv, index=False)




#%%
#################################################
# Check our dataset
#################################################

df.head()

print(f"Total Files: {len(df)}")
print(f"Unique File Extensions: {df['Extension'].unique()}")
print(f"Unique Duration: {df['Duration'].unique()}")
print(f"Min Duration: {df['Duration'].min()}")
print(f"Max Duration: {df['Duration'].max()}")
print(f"Unique Channels: {df['Channels'].unique()}")
print(f"Unique SampleRate: {df['SampleRate'].unique()}")

# Check the distribution of classes
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize = (5, 10))
sns.countplot(y = df['class'], palette = 'viridis')
plt.title('Distribution of Classes', fontsize = 16)
plt.xlabel('Count', fontsize = 14)
plt.ylabel('Class', fontsize = 14)
plt.show()

# Check the distribution of duration
plt.figure(figsize = (5, 10))
sns.countplot(y = round(df['Duration'],4), palette = 'viridis')
plt.title('Distribution of Classes', fontsize = 16)
plt.xlabel('Count', fontsize = 14)
plt.ylabel('Class', fontsize = 14)
plt.show()



#%%
############################################################################
############################################################################
############################################################################

from tqdm import trange,tqdm
import torch.optim as optim
from classification_dataset import SoundCustomDS
import torch
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
#from classification_model import AlexNet, CNN, EarlyStopping
import torchvision.models as models
from plot_results import plot_confusion_matrix, plot_validation_accuracy, plot_validation_loss
import optuna
from sklearn.metrics import confusion_matrix
from torchsummary import summary 
from torchview import draw_graph 
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import graphviz
import datetime
import warnings
import ast
pd.set_option('display.float_format', '{:.8f}'.format)
warnings.filterwarnings("ignore")

df_train = df[df['train_test'] == 'train'].reset_index()
df_test = df[df["train_test"]=="test"].reset_index()

duration = 5000
sr = 44100
channel = 1

# Set random seed for reproducibility
torch.manual_seed(145)  # Set seed for CPU operations
if torch.cuda.is_available():
    torch.cuda.manual_seed(145)
    
# Create the dataset
mydataset = SoundCustomDS(df_train, duration, sr, channel)
        
# Random split of 80:20 between training and validation
num_items = len(mydataset)
num_train = round(num_items * 0.8)
num_val = num_items - num_train
train_ds, val_ds = random_split(mydataset, [num_train, num_val])


# Check the shape of the validation dataloader
#for X, y in val_dl:
#    print(f"Shape of X [N, C, H, W]: {X.shape}")
#    print(f"Shape of y: {len(y)}")
#    break
    

# Print model summary  
#os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'
#summary(model, input_size=(1, 64, 860))
#input_data = torch.randn(64, 1, 64, 860)
#model_graph = draw_graph(model,input_data, roll=True)
#model_graph.visual_graph


def train(model, device, train_loader, optimizer, epoch, loss_criteria):
    # Set the model to training mode
    model.train()
    train_loss = 0
    correct = 0
    total_prediction = 0
    print("------------------------------------ Epoch:", epoch,"------------------------------------")
    # Process the images in batches
    for batch_idx, (data, target) in enumerate(train_loader):
        # Use the CPU or GPU as appropriate
        # Recall that GPU is optimized for the operations we are dealing with
        data, target = data.to(device), target.to(device)

        # Reset the optimizer
        optimizer.zero_grad()

        # Push the data forward through the model layers
        output = model(data.to(device))

        # Get the loss
        loss = loss_criteria(output, target)

        # Keep a running total
        train_loss += loss.item()
        
        # Get the predicted class with the highest score
        _, predicted = torch.max(output.data, 1)
        # Calculate the accuracy for this batch
        correct += torch.sum(target==predicted).item()
        total_prediction += predicted.shape[0]
            
        # Backpropagate
        loss.backward(retain_graph=True)
        optimizer.step()

        # Print metrics so we see some progress
        #print('\tTraining batch {} Loss: {:.6f}'.format(batch_idx + 1, loss.item()))
            
    # return average loss for the epoch
    avg_loss = train_loss / (batch_idx+1)
    avg_acc = correct / total_prediction
    print('Training set: Average loss: {:.6f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        avg_loss, correct, len(train_loader.dataset),
        100. * correct / len(train_loader.dataset)))
    return avg_loss, avg_acc


def validate(model, device, test_loader, loss_criteria):
    # Switch the model to evaluation mode (so we don't backpropagate or drop)
    model.eval()
    test_loss = 0
    correct = 0
    total_prediction = 0
    with torch.no_grad():
        batch_count = 0
        for data, target in test_loader:
            batch_count += 1
            data, target = data.to(device), target.to(device)
            
            # Get the predicted classes for this batch
            output = model(data)
            
            # Calculate the loss for this batch
            test_loss += loss_criteria(output, target).item()
            
            # Get the predicted class with the highest score
            _, predicted = torch.max(output.data, 1)
            # Calculate the accuracy for this batch
            correct += torch.sum(target==predicted).item()
            total_prediction += predicted.shape[0]
            
    # Calculate the average loss and total accuracy for this epoch
    avg_loss = test_loss / batch_count
    avg_acc = correct / total_prediction
    print('Validation set: Average loss: {:.6f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        avg_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
    # return average loss for the epoch
    return avg_loss, avg_acc


def train_val(model, device, epochs, lr, batch_size, train_dl, val_dl):
    # Instantiate the EarlyStopping class
    #early_stopping = EarlyStopping(patience=5)
    early_stopping_patience = 5
    best_val_acc = 0.0
    epochs_no_improve = 0
    
    # Define loss function and optimizer
    loss_criteria = nn.CrossEntropyLoss()
    #optimizer = optim.Adam(model.parameters(), lr=lr)
    # Only optimize the parameters that are not frozen
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    
    # Track metrics in these arrays
    train_val_results = []
    
    for epoch in tqdm(range(1, epochs + 1)):
        start_time = datetime.datetime.now()
        train_loss, train_acc = train(model, device, train_dl, optimizer, epoch, loss_criteria)
        val_loss, val_acc = validate(model, device, val_dl, loss_criteria)
        end_time = datetime.datetime.now()
        train_val_results.append([epoch, lr, batch_size, train_loss, val_loss, train_acc, val_acc, start_time, end_time])
        
        # Report intermediate results to Optuna
        #trial.report(val_loss, epoch)
        
        #early_stopping(val_loss, model)
        #if early_stopping.early_stop:
        #    print("Early stopping")
        #    break  
        
        if val_acc > best_val_acc:
            print(f'Validation accuracy increased ({best_val_acc:.6f} --> {val_acc:.6f}).')
            best_val_acc = val_acc
            #best_model_state_dict = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f'EarlyStopping counter: {epochs_no_improve} out of {early_stopping_patience}')

        if epochs_no_improve == early_stopping_patience:
            print("Early stopping!")
            break
    return val_acc, train_val_results, best_val_acc#, best_model_state_dict


   
def objective(trial):
    # Define hyperparameters to optimize
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    epochs = 100
    
    # Define the model
    num_classes=50
    # Load pre-trained ResNet18 model
    model = models.resnet18(pretrained=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Modify the first convolutional layer to accept single-channel (64x860) input
    model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    # Modify the final fully connected layer to output 50 classes (ESC-50 has 50 categories)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze the parameters of the last two layers
    #for param in model.layer4.parameters():
    #    param.requires_grad = True
    for param in model.fc.parameters():
        param.requires_grad = True
    
    print('***********************************************************************************')
    print(f'Trial number: {trial.number + 1}')
    print(f'Hyperparameters ----> Epochs: {epochs},   LR: {lr},   Batch Size: {batch_size}')
    print('***********************************************************************************')
        
    # Create training and validation data loaders
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, worker_init_fn=np.random.seed(145))
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, worker_init_fn=np.random.seed(145))
        
    val_acc, train_val_results, best_val_acc  = train_val(model, device, epochs, lr, batch_size, train_dl, val_dl)    
    trial_results_df = pd.DataFrame(train_val_results, columns = ['epoch', 'lr', 'batch_size', 'train_loss', 'val_loss', 'train_acc', 'val_acc', 'start_time', 'end_time'])
    trial_results_df.insert(0, 'trial', trial.number + 1)
    
    objective.df_train_val_results = pd.concat([objective.df_train_val_results, trial_results_df], ignore_index=True)
    
    # Save the model if it's the best model so far
    if objective.best_acc is None or best_val_acc > objective.best_acc:
        print(f'This is the best model so far --> Current Val Accuracy: {best_val_acc}  -  Previous Accuracy {objective.best_acc}')
        objective.best_acc = best_val_acc
        print('Model is saved....')
        torch.save(model, models_directory + model_name + "_best_model.pth")
            
        # Implement early stopping based on validation loss
        #if trial.should_prune():
        #    raise optuna.exceptions.TrialPruned()
        
    return val_acc   

objective.df_train_val_results = pd.DataFrame(columns = ['trial', 'epoch', 'lr', 'batch_size', 'train_loss', 'val_loss', 'train_acc', 'val_acc', 'start_time', 'end_time'])
objective.best_acc = None

# Create an Optuna study object
study = optuna.create_study(direction="maximize")

# Run the optimization
study.optimize(objective, n_trials=10)

# Export Results to csv
objective.df_train_val_results['total_time_seconds'] = (objective.df_train_val_results['end_time'] - objective.df_train_val_results['start_time']).dt.total_seconds()
objective.df_train_val_results.to_csv(models_directory + model_name + '.csv', index=False)

# Access the best hyperparameters found by Optuna
best_params = study.best_params
print(f'The best hyperparameters are: {best_params}')


##############################################################################
##########################      TEST THE MODEL      ##########################
##############################################################################

# Initialize the best model
model = torch.load(models_directory + model_name + "_best_model.pth")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
df = pd.read_csv(dataframe_csv)
df_test = df[df["train_test"]=="test"].reset_index()
model_params = {}
with open(models_directory + model_name + '_parameters.txt', 'r') as file:
    file_content = file.read()
    model_params = ast.literal_eval(file_content)
            
test_ds = mydataset = SoundCustomDS(df_test, model_params['duration'], model_params['sr'], model_params['channel'])
test_dl = DataLoader(test_ds, batch_size=model_params['batch_size'], shuffle=False)


# Test the model
model.eval()
test_correct = 0
test_total = 0
true_labels = []
predicted_labels = []
with torch.no_grad():
    for inputs, labels in test_dl:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()
        true_labels.extend(labels.cpu().numpy())
        predicted_labels.extend(predicted.cpu().numpy())
test_accuracy = 100.0 * test_correct / test_total
print(f'Test Accuracy: {test_accuracy:.2f}%')


# Plot the Confusion Matrix
plot_confusion_matrix(df_test, true_labels, predicted_labels)


# Export the probabilities per class. This is to be able to evaluate if there are classes that are predicted with high confidence 
# or classes closed to each other.
class_names = df_test['class'].unique()
class_names = sorted(class_names)

def predict(model, dataloader, device, true_labels):
    model.eval()
    all_probs = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probs  = F.softmax(outputs, dim=1)
            #_, predicted = torch.max(outputs.data, 1)
            all_probs.extend(probs.cpu().numpy())
            
    # Convert the list of probabilities into a DataFrame
    probs_df = pd.DataFrame(all_probs, columns=class_names)
    
    # Get the predicted class for each sample
    predicted_labels = probs_df.idxmax(axis=1)

    # Calculate accuracy
    accuracy = accuracy_score(true_labels, predicted_labels)
    print(f"Accuracy: {accuracy:.4f}")
    
    return all_probs, predicted_labels


# Get the top classes that are closer to each other to a specific threshold 
def get_top_classes_within_threshold(row, threshold=0.2):
    top_prob = row.max()
    within_threshold = row[row >= top_prob * (1 - threshold)].sort_values(ascending=False).index.tolist()
    return within_threshold

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

true_labels = df_test['class'].values
probabilities, predicted_labels = predict(model, test_dl, device, true_labels)

probs_df = pd.DataFrame(probabilities, columns=class_names)
predicted_labels = probs_df.idxmax(axis=1)

top_classes_within_20_percent = probs_df.apply(get_top_classes_within_threshold, axis=1)

# Create the final DataFrame
probabilities_df = pd.concat([df_test[['File Name', 'class']].reset_index(drop=True), 
                      predicted_labels.rename('predicted_class'), 
                      probs_df, 
                      top_classes_within_20_percent.rename('top_classes_within_20_percent')], axis=1)

# Save the dataframe
probabilities_df.to_csv(models_directory  + model_name + '_probabilities.csv', index=False)

