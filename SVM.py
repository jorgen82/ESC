import pandas as pd
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV, RepeatedStratifiedKFold
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Do we have already exported the mfcc values or not?
# Load the model or train?
read_mfcc_csv = True
load_model = True

# Path initialization
root_path = 'C:/Users/george.apostolakis/Downloads/ESC-50/'
models_path = root_path + 'models/'
model_file = models_path + "svm_model.pkl"
dataframe_csv = root_path + 'dataframe.csv'
mfcc_csv = root_path + 'dataframe_mfcc.csv'


def extract_mfcc(file_path):
    # Load audio file
    audio, sample_rate = librosa.load(file_path, sr=44100, mono=True, duration=5.0)

    # Pad or truncate to 5 seconds
    target_length = 5 * sample_rate
    if len(audio) < target_length:
        # Pad with zeros
        audio = np.pad(audio, (0, target_length - len(audio)))
    elif len(audio) > target_length:
        # Truncate
        audio = audio[:target_length]

    # Extract MFCC features
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=num_mfcc)
    return mfccs.mean(axis=1)  # Taking mean of MFCC coefficients

global num_mfcc 
num_mfcc = 15

if read_mfcc_csv != True:
    df = pd.read_csv(dataframe_csv)
    df['file_path'] = df['Subfolder_Path'] + '/' + df['File Name']
    
    df['mfcc'] = df['file_path'].apply(extract_mfcc)
    mfcc_df = pd.DataFrame(df['mfcc'].tolist())

    scaler = StandardScaler()
    mfcc_df = pd.DataFrame(scaler.fit_transform(mfcc_df))

    mfcc_df.columns = [f'mfcc_{i}' for i in range(len(mfcc_df.columns))]
    df = pd.concat([df, mfcc_df], axis=1)

    df.to_csv(mfcc_csv, index=False)
else:
    df = pd.read_csv(mfcc_csv)


df_train = df[df['train_test'] == 'train']
df_test = df[df["train_test"]=="test"]

columns_to_keep = ['label'] + [f'mfcc_{i}' for i in range(num_mfcc)]
#df_train = df_train[columns_to_keep]
#df_test = df_test[columns_to_keep]



train_array = df_train[columns_to_keep].values
X_part = train_array[:,1:df_train[columns_to_keep].shape[1]]
y_part = train_array[:,0]

X_train_val, X_test_val, y_train_val, y_test_val = train_test_split(X_part, y_part, test_size=0.1,random_state=8)

test_array = df_test[columns_to_keep].values
X_test = test_array[:,1:df_test[columns_to_keep].shape[1]]
y_test = test_array[:,0]

def svm_model(X_train_val, y_train_val, X_test_val, y_test_val, X_test, y_test):
    start_time = datetime.datetime.now()
    # Define the hyperparameters
    kernel = ['poly', 'rbf', 'sigmoid']
    C = [50, 10, 1.0, 0.1, 0.01]
    gamma = [1, 0.1, 0.01, 0.001]
    
    model_svm = SVC()

    # Define grid search
    param_grid = dict(kernel=kernel, C=C, gamma=gamma)

    # We will use the RepeatedStratifiedKFold as a cross validation. Its usefull in classification tasks with inbalanced data
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    grid_svm = GridSearchCV(model_svm, param_grid=param_grid, cv=cv, refit=True, scoring='accuracy')

    # Fit the model
    grid_svm.fit(X_train_val, y_train_val)

    # Best parameters and accuracy
    print("SVM Best parameters found: ", grid_svm.best_params_)
    print("SVM Best cross-validation accuracy: ", grid_svm.best_score_)

    best_model_svm = grid_svm.best_estimator_
    joblib.dump(best_model_svm, model_file)

    # Evaluate on the validation set
    best_model_svm = grid_svm.best_estimator_
    val_predictions = best_model_svm.predict(X_test_val)
    val_accuracy = accuracy_score(y_test_val, val_predictions)
    print("SVM Validation accuracy: ", val_accuracy)

    # Evaluate the model on the test set
    test_predictions = best_model_svm.predict(X_test)
    test_accuracy = accuracy_score(y_test, test_predictions)
    print("SVM Test accuracy: ", test_accuracy)
    end_time = datetime.datetime.now()
    time_difference = end_time - start_time
    hours_difference = time_difference.total_seconds() / 3600
    print(f'"SVM Train time in hours:" {hours_difference:.2f}')
    return test_predictions

def randomforest_model(X_train_val, y_train_val, X_test_val, y_test_val, X_test, y_test):
    start_time = datetime.datetime.now()
    # Define the hyperparameters
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_features': ['auto', 'sqrt', 'log2'],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }

    model_rf = RandomForestClassifier(random_state=42)
    
    # We will use the RepeatedStratifiedKFold as a cross validation. Its usefull in classification tasks with inbalanced data
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    grid_rf = GridSearchCV(model_rf, param_grid=param_grid, cv=cv, refit=True, scoring='accuracy')

    # Fit the model
    grid_rf.fit(X_train_val, y_train_val)

    # Best parameters and accuracy
    print("Random Forest Best parameters found: ", grid_rf.best_params_)
    print("Random Forest Best cross-validation accuracy: ", grid_rf.best_score_)

    best_model_rf = grid_rf.best_estimator_
    joblib.dump(best_model_rf, models_path + "rf_model.pkl")

    # Evaluate on the validation set
    best_model_rf = grid_rf.best_estimator_
    val_predictions = best_model_rf.predict(X_test_val)
    val_accuracy = accuracy_score(y_test_val, val_predictions)
    print("Random Forest Validation accuracy: ", val_accuracy)

    # Evaluate the model on the test set
    test_predictions = best_model_rf.predict(X_test)
    test_accuracy = accuracy_score(y_test, test_predictions)
    print("Random Forest Test accuracy: ", test_accuracy)
    end_time = datetime.datetime.now()
    time_difference = end_time - start_time
    hours_difference = time_difference.total_seconds() / 3600
    print(f'"Random Forest Train time in hours:" {hours_difference:.2f}')
    return test_predictions
    
if load_model:
    model = joblib.load(model_file)
    y_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    print("SVM Test accuracy: ", test_accuracy)
else:
    y_pred = svm_model(X_train_val, y_train_val, X_test_val, y_test_val, X_test, y_test)

#y_pred = randomforest_model(X_train_val, y_train_val, X_test_val, y_test_val, X_test, y_test)

# Generate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix
mask = conf_matrix == 0
annot = np.where(mask, '', conf_matrix)

plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=annot, fmt="", cmap="Blues", xticklabels=df_test['class'].unique(), yticklabels=df['class'].unique())
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Identify incorrectly classified samples
label_to_class = dict(zip(df_test['label'], df_test['class']))

incorrect_indices = np.where(y_test != y_pred)[0]
incorrect_classifications = df.iloc[incorrect_indices][['File Name', 'class']]
incorrect_classifications['predicted_class'] = y_pred[incorrect_indices]
incorrect_classifications['predicted_class'] = incorrect_classifications['predicted_class'].map(label_to_class)
# Display the list of incorrectly classified samples
print(incorrect_classifications)

df.to_csv(models_path + 'incorrect_predictions.csv', index=False)

