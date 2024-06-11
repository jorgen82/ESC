#%%
import pandas as pd
from initialize import file_exploration, combine_audio_visuals, data_augmentation, FileMover
from sklearn import preprocessing 
pd.set_option('display.float_format', '{:.8f}'.format)

# Create or Read dataset
read_dataset = True
export_audio_visuals = True
augment_data = False

# Path initialization
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
    # Augment audio files
    if augment_data: data_augmentation(source_directory) 
    # Fetch audio file info and export the visuals (if export_audio_visuals = True)
    df = pd.DataFrame(file_exploration(source_dir = source_directory, output_root_folders = output_root_folders, export_audio_visuals = export_audio_visuals), columns=['Subfolder', 'Subfolder_Path', 'File Name', 'Extension', 'Duration', 'Channels', 'SampleRate'])
    df['class'] = df['Subfolder'].astype(str).str[6:]
    label_encoder = preprocessing.LabelEncoder() 
    df['label']= label_encoder.fit_transform(df['class']) 
    # Combine visuals
    combine_audio_visuals(comboplots_output_root_folder, root_path)
    # Move test data to the test_files_directory
    subset_percentage = 0.2
    df['train_test'] = 'train'
    file_mover = FileMover(df, source_directory, test_files_directory, subset_percentage)
    file_mover.move_files_with_subset()

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
from classification_model import AlexNet, CNN, EarlyStopping, CombinedModel
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
warnings.filterwarnings("ignore")

df_train = df[df['train_test'] == 'train'].reset_index()

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
#val_dl = DataLoader(val_ds, batch_size=64, shuffle=False, worker_init_fn=np.random.seed(145))
#for X, y in val_dl:
#    print(f"Shape of X [N, C, H, W]: {X.shape}")
#    print(f"Shape of y: {len(y)}")
#    break
    
# Create the model and put it on the GPU if available
#num_classes=50
#model = CombinedModel(num_classes)
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#model = model.to(device)
#next(model.parameters()).device
   
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
    #optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.001)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), weight_decay=0.001, lr=lr)
    
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
    model = CombinedModel(num_classes)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

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
        torch.save(model, models_directory + "CNN_best_model.pth")
            
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
objective.df_train_val_results.to_csv(models_directory + 'CNN.csv', index=False)

# Access the best hyperparameters found by Optuna
best_params = study.best_params
print(f'The best hyperparameters are: {best_params}')


# Plot Validation Accuracy and Loss visuals
plot_validation_accuracy(objective.df_train_val_results)
plot_validation_loss(objective.df_train_val_results)



##############################################################################
##########################      TEST THE MODEL      ##########################
##############################################################################

# Initialize the best model
model = torch.load(models_directory + 'CNN_best_model.pth')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
df = pd.read_csv(dataframe_csv)
df_test = df[df["train_test"]=="test"].reset_index()
model_params = {}
with open(models_directory + '/Transfer Learning/resnet_melgram_1_64x860_parameters.txt', 'r') as file:
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


# Export and prind the top 4 propabilities per class

model.eval()
total_correct = 0
total_samples = 0

# List to store detailed results for each input
detailed_results = []
results = []
class_names = df_test['class'].unique().tolist()

with torch.no_grad():
    for batch in test_dl:
        waveforms, labels = batch
        outputs = model(waveforms)
        _, predicted = torch.max(outputs, 1)

        # Compute probabilities using softmax
        probabilities = F.softmax(outputs, dim=1)

        # Get top 4 classes and their probabilities
        top4_probs, top4_indices = torch.topk(probabilities, 4, dim=1)

        # Calculate total correct predictions
        total_correct += (predicted == labels).sum().item()
        total_samples += labels.size(0)

        # Process each input in the batch
        for i in range(len(waveforms)):
            file_info = df_test.iloc[total_samples - len(waveforms) + i]
            file_name = file_info['File Name']
            true_class = file_info['class']
            true_label = file_info['label']
            
            top4_predictions = top4_indices[i].tolist()
            top4_probabilities = top4_probs[i].tolist()
            top4_classes = [df_test[df_test['label'] == idx]['class'].values[0] for idx in top4_predictions]

            detailed_results.append({
                'fileName': file_name,
                'true_class': true_class,
                'top_4_classes': top4_classes,
                'top_4_probabilities': top4_probabilities
            })
            
            # Store the probabilities for each class
            result = {'fileName': file_name, 'true_class': true_class}
            for j, class_name in enumerate(class_names):
                result[class_name] = probabilities[i][j]
            results.append(result)

# Calculate overall accuracy
accuracy = total_correct / total_samples
print(f"Total Accuracy: {accuracy:.4f}")

# Create a dataframe with the results
probabilities_df = pd.DataFrame(results)
probabilities_df.to_csv(models_directory + 'probabilities.csv', index=False)

# Display detailed results
for result in detailed_results:
    print(f"FileName: {result['fileName']}, True Class: {result['true_class']}")
    for j in range(4):
        print(f"  Top {j + 1}: {result['top_4_classes'][j]} - Probability: {result['top_4_probabilities'][j]:.4f}")
    print()
