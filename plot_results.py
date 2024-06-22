"""
    Plot some visuals 
    1) plot_confusion_matrix
    2) plot_validation_accuracy
    3) plot_validation_loss
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


# Plot the confusion matrix
def plot_confusion_matrix(df_test, true_labels, predicted_labels):
    # Create a mapping from encoded labels to class names
    label_class = df_test[['label' ,'class']].drop_duplicates()
    class_mapping = dict(zip(label_class['label'], label_class['class']))


    # Convert encoded labels to class names
    true_class_labels = np.array([class_mapping[label] for label in true_labels])
    predicted_class_labels = np.array([class_mapping[label] for label in predicted_labels])

    # Extract unique class names from the DataFrame
    class_names = df_test['class'].unique()

    # Compute confusion matrix
    cm = confusion_matrix(true_class_labels, predicted_class_labels, labels=class_names)

    # Create a DataFrame for better visualization
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)

    # Plotting the confusion matrix
    mask = cm == 0
    annot = np.where(mask, '', cm)
    plt.figure(figsize=(15, 12))
    sns.heatmap(cm_df, annot=annot, fmt="", cmap='Blues')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.show()
 
    
# Plot the Accuracy
def plot_validation_accuracy(df):
    
    grouped = df.groupby('trial')

    best_trial = df.loc[df['val_acc'].idxmax(), 'trial']

    # Plot subplots with 2 plots per row
    fig, axs = plt.subplots(len(grouped) // 2 + len(grouped) % 2, 2, figsize=(12, 6 * (len(grouped) // 2 + len(grouped) % 2)))

    for i, (trial, group) in enumerate(grouped):
        row = i // 2
        col = i % 2
        ax = axs[row, col] if len(grouped) > 1 else axs
        
        ax.plot(group['epoch'], group['train_acc'], label='Train Accuracy')
        ax.plot(group['epoch'], group['val_acc'], label='Validation Accuracy')
        
        # Find index of minimum val_acc
        max_val_acc_idx = group['val_acc'].idxmax()
        
        # Plot red dot for minimum val_acc
        ax.plot(group.loc[max_val_acc_idx, 'epoch'], group.loc[max_val_acc_idx, 'val_acc'], 'ro')
        
        # Show minimum val_acc, lr, and batch_size on bottom right of each subplot
        min_val_acc = group.loc[max_val_acc_idx, 'val_acc']
        lr = group.loc[max_val_acc_idx, 'lr']
        batch_size = group.loc[max_val_acc_idx, 'batch_size']
        ax.annotate(f'Max Val Acc: {min_val_acc:.4f}\nLR: {lr}\nBatch Size: {batch_size}',
                    xy=(1, 0), xycoords='axes fraction',
                    xytext=(-5, 5), textcoords='offset points',
                    ha='right', va='bottom')
        
        # Add (best trial) to the subplot title if it's the best trial
        if trial == best_trial:
            ax.set_title(f'Trial {trial} (best trial)')
        else:
            ax.set_title(f'Trial {trial}')
        
        ax.set_ylabel('Accuracy')
        ax.grid(True)
        ax.legend()

    # Set common x-axis label
    ax.set_xlabel('Epoch')

    plt.tight_layout()
    plt.show()
    return



# Plot the Validation Loss
def plot_validation_loss(df):
    grouped = df.groupby('trial')

    best_trial = df.loc[df['val_loss'].idxmin(), 'trial']

    # Plot subplots with 2 plots per row
    fig, axs = plt.subplots(len(grouped) // 2 + len(grouped) % 2, 2, figsize=(12, 6 * (len(grouped) // 2 + len(grouped) % 2)))

    for i, (trial, group) in enumerate(grouped):
        row = i // 2
        col = i % 2
        ax = axs[row, col] if len(grouped) > 1 else axs
        
        ax.plot(group['epoch'], group['train_loss'], label='Train Loss')
        ax.plot(group['epoch'], group['val_loss'], label='Validation Loss')
        
        # Find index of minimum val_loss
        min_val_loss_idx = group['val_loss'].idxmin()
        
        # Plot red dot for minimum val_loss
        ax.plot(group.loc[min_val_loss_idx, 'epoch'], group.loc[min_val_loss_idx, 'val_loss'], 'ro')
        
        # Show minimum val_loss, lr, and batch_size on bottom right of each subplot
        min_val_loss = group.loc[min_val_loss_idx, 'val_loss']
        lr = group.loc[min_val_loss_idx, 'lr']
        batch_size = group.loc[min_val_loss_idx, 'batch_size']
        ax.annotate(f'Min Val Loss: {min_val_loss:.4f}\nLR: {lr}\nBatch Size: {batch_size}',
                    xy=(1, 0), xycoords='axes fraction',
                    xytext=(-5, 5), textcoords='offset points',
                    ha='right', va='bottom')
        
        # Add (best trial) to the subplot title if it's the best trial
        if trial == best_trial:
            ax.set_title(f'Trial {trial} (best trial)')
        else:
            ax.set_title(f'Trial {trial}')
        
        ax.set_ylabel('Loss')
        ax.grid(True)
        ax.legend()

    # Set common x-axis label
    ax.set_xlabel('Epoch')

    plt.tight_layout()
    plt.show()