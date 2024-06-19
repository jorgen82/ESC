"""
    Creates the classification models which we will use on our experiments
    1) CNN: This is a custom cnn model
    2) CNN_square: This is for testing the 860x860 melgrams rather than the 64x860 which is the original size - dataset should be altered to contain relevant processing function
    3) CombinedModel: This is a combination of pretrained Resnet18 with a custom CNN (defined on the CustomCNN class)
    4) VGGishClassifier: Pretrained VGGish model. - Did not use this on our final experiments.
    5) EarlyStopping: A class for early stopping of the training (if its not improving for 5 consecutive epochs) - Did not use this on our final experiments.
"""

import torch
import torch.nn as nn
import numpy as np
import torchvision.models as models
import torch.nn.functional as F



# ----------------------------
# Audio Classification Model
# ----------------------------
class CNN (nn.Module):
    # ----------------------------
    # Build the model architecture
    # ----------------------------
    def __init__(self, num_classes=50):
        super().__init__()
        
        self.conv1=nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=8,kernel_size=3,stride=2,padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2=nn.Sequential(
            nn.Conv2d(in_channels=8,out_channels=16,kernel_size=3,stride=2,padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv3=nn.Sequential(
            nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv4=nn.Sequential(
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2)
        ) 
        self.conv5=nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128)#,
            #nn.MaxPool2d(kernel_size=2)
        ) 
        self.connected_layer=nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.25),
            #nn.Linear(in_features=576,out_features=288),
            nn.Linear(in_features=1664,out_features=1024),
            nn.Dropout(0.25),
            nn.ReLU(),
            nn.Linear(in_features=1024,out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512,out_features=num_classes),
            #nn.Dropout(0.25),
            #nn.Linear(in_features=288,out_features=num_classes),
            #nn.ReLU(),
            #nn.Linear(in_features=512,out_features=num_classes),
            nn.Softmax(dim=1)
        )

    # ----------------------------
    # Forward pass computations
    # ----------------------------
    def forward(self, x):
        x=self.conv1(x)
        x=self.conv2(x)
        x=self.conv3(x)
        x=self.conv4(x)
        x=self.conv5(x)
        #x = x.view(x.size(0), -1).size(1)
        #print("Size of the flattened tensor:", x)
        x = self.connected_layer(x)
        #x = F.softmax(x, dim=1)
        return x

  
  
    
class CNN_square (nn.Module):
    # ----------------------------
    # Build the model architecture
    # ----------------------------
    def __init__(self, num_classes=50):
        super().__init__()
        
        self.conv1=nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=8,kernel_size=3,stride=2,padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2=nn.Sequential(
            nn.Conv2d(in_channels=8,out_channels=16,kernel_size=3,stride=2,padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv3=nn.Sequential(
            nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3,stride=2,padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv4=nn.Sequential(
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,stride=2,padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2)
        ) 
        self.connected_layer=nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.25),
            nn.Linear(in_features=576,out_features=288),
            nn.ReLU(),
            #nn.Dropout(0.25),
            nn.Linear(in_features=288,out_features=num_classes),
            #nn.ReLU(),
            #nn.Linear(in_features=512,out_features=num_classes),
            nn.Softmax(dim=1)
        )

    # ----------------------------
    # Forward pass computations
    # ----------------------------
    def forward(self, x):
        x=self.conv1(x)
        x=self.conv2(x)
        x=self.conv3(x)
        x=self.conv4(x)
        #x = x.view(x.size(0), -1).size(1)
        #print("Size of the flattened tensor:", x)
        x = self.connected_layer(x)
        return x


class CustomCNN(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(CustomCNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        #self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        #self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)

        self.fc1 = nn.Linear(256 , 128)  # Adjust based on input size after convolutions and pooling
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        #print("Input shape:", x.shape)
        x = F.relu(self.bn1(self.conv1(x)))
        #print("After layer 1 shape:", x.shape)
        x = F.relu(self.bn2(self.conv2(x)))
        #print("After layer 2 shape:", x.shape)
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        #print("After layer 3 shape:", x.shape)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        #print("Flattened:", x.shape )
        x = F.relu(self.fc1(x))
        #print("After fc1 shape:" ,x.shape)
        x = self.fc2(x)
        #print("After fc2 shape:" ,x.shape)
        return x


class CombinedModel(nn.Module):
    def __init__(self, num_classes):
        super(CombinedModel, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        
        # Freeze all layers in ResNet
        for param in self.resnet.parameters():
            param.requires_grad = False
        
        # Unfreeze layer4
        for param in self.resnet.layer4.parameters():
            param.requires_grad = True
                
        # Modify ResNet to accept single-channel (mel-spectrogram) input
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.fc = nn.Identity()  # Remove the final fully connected layer

        # Custom CNN expects 512 channels from the ResNet output
        self.custom_cnn = CustomCNN(input_channels=512, num_classes=num_classes)

    def forward(self, x):
        #print("Resnet input shape:", x.shape)
        x = self.resnet(x)  # Output shape: (batch_size, 512, H, W)
        #print("After resnet shape:", x.shape)
        x = x.unsqueeze(2).unsqueeze(3)  # Ensure the tensor has shape (batch_size, 512, height, width)
        #print("After unsqueeze shape:", x.shape)
        x = self.custom_cnn(x)
        return x

    
class VGGishClassifier(nn.Module):
    def __init__(self, base_model, num_classes=2):
        super(VGGishClassifier, self).__init__()
        self.features = base_model.features
        #self.embeddings = base_model.embeddings
        self.embeddings = nn.Sequential(
            nn.Linear(512 * 4 * 53, 4096),  # Adjust the input size based on feature map size
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 128),
            nn.ReLU(inplace=True)
        )
        
        # Freeze pre-trained layers
        for param in self.features.parameters():
            param.requires_grad = False
        for param in self.embeddings.parameters():
            param.requires_grad = True
        
        # Add a new classifier layer
        self.classifier = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)                # Pass input through feature extractor
        #print(f'Feature shape: {x.shape}')  # Debug: Check the shape after features
        x = x.view(x.size(0), -1)           # Flatten the tensor
        #print(f'Flattened shape: {x.shape}')# Debug: Check the shape after flattening
        x = self.embeddings(x)              # Pass through embeddings
        #print(f'Embeddings shape: {x.shape}')# Debug: Check the shape after embeddings
        x = self.classifier(x)              # Pass through classifier
        return x
    


class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf  # Initialize with positive infinity

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score <= self.best_score - self.delta:  # Modify condition to use delta
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if val_loss < self.val_loss_min:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
            torch.save(model.state_dict(), 'checkpoint.pt')
            self.val_loss_min = val_loss
            
