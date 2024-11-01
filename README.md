## Sleep Efficiency Prediction using a Feed-Forward Neural Network
This project predicts sleep efficiency based on various factors, including age, sleep duration, REM sleep percentage, and lifestyle habits. The model is implemented as a feed-forward neural network built with TensorFlow and Keras.

# Dataset
The dataset used in this project contains several features related to sleep patterns, demographics, and lifestyle. Each row represents an individual's sleep data, including the following key features:

Age: The age of the individual.
Gender: Gender of the individual.
Sleep Duration: Number of hours spent sleeping.
REM Sleep Percentage: Percentage of sleep spent in REM.
Deep Sleep Percentage: Percentage of sleep spent in deep sleep.
Light Sleep Percentage: Percentage of sleep spent in light sleep.
Awakenings: Number of times the individual woke up during the night.
Caffeine Consumption: Amount of caffeine consumed.
Alcohol Consumption: Amount of alcohol consumed.
Smoking Status: Whether the individual smokes.
Exercise Frequency: Frequency of exercise.
The target variable is Sleep Efficiency, which measures how efficiently the individual sleeps.

# Dataset Source
You can find various sleep datasets on platforms like Kaggle or Hugging Face Datasets. Alternatively, you can upload a compatible dataset directly for this project.

# Model Architecture
The model is a simple feed-forward neural network with the following structure:

Input Layer: Receives multiple input features from the dataset.
Hidden Layer 1: A dense layer with 64 neurons and ReLU activation.
Hidden Layer 2: A dense layer with 32 neurons and ReLU activation.
Output Layer: A single neuron for predicting sleep efficiency (as this is a regression task).
The loss function used is mean squared error (MSE), and the optimizer used is Adam with a learning rate of 0.001.

# Code Structure and Usage
The project is organized into modules for easy use and reusability:

1. dataset_module.py
This module contains the load_and_preprocess_data function, which handles data loading, preprocessing, and scaling.

# dataset_module.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_and_preprocess_data(file_path):
    """
    Loads and preprocesses the dataset for training.

    Parameters:
    - file_path (str): Path to the CSV file.

    Returns:
    - X_train_scaled, X_test_scaled, y_train, y_test: Processed and scaled training and test data.
    """
    data = pd.read_csv(file_path)
    data_cleaned = data.drop(columns=["ID", "Bedtime", "Wakeup time"]).dropna()
    
   
    label_encoder = LabelEncoder()
    data_cleaned['Gender'] = label_encoder.fit_transform(data_cleaned['Gender'])
    data_cleaned['Smoking status'] = label_encoder.fit_transform(data_cleaned['Smoking status'])

    
    X = data_cleaned.drop(columns=["Sleep efficiency"]).values
    y = data_cleaned["Sleep efficiency"].values

    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test

# model_module.py

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

class NeuralNetworkModel:
    def __init__(self, input_dim, learning_rate=0.001):
        self.model = Sequential([
            Dense(64, input_dim=input_dim, activation='relu'),
            Dense(32, activation='relu'),
            Dense(1)  # Output layer for regression
        ])
        self.model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=learning_rate))
    
    def train(self, X_train, y_train, X_val, y_val, epochs=20, batch_size=10):
        return self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val))
    
    def evaluate(self, X_test, y_test):
        test_loss = self.model.evaluate(X_test, y_test)
        print(f"Test Loss (Mean Squared Error): {test_loss}")
        return test_loss

3. Jupyter Notebook (Sleep_Efficiency_Prediction_Notebook.ipynb)
This notebook demonstrates how to use the modules to load data, initialize the model, train, and evaluate it.

Instructions for Running the Code
Install Dependencies:

Ensure you have the necessary libraries installed:
pip install tensorflow scikit-learn pandas
Using the Modules:

Dataset Module:


from dataset_module import load_and_preprocess_data
X_train_scaled, X_test_scaled, y_train, y_test = load_and_preprocess_data('Sleep_Efficiency.csv')
Model Module:
python

from model_module import NeuralNetworkModel
model = NeuralNetworkModel(input_dim=X_train_scaled.shape[1])
history = model.train(X_train_scaled, y_train, X_test_scaled, y_test, epochs=20, batch_size=10)
model.evaluate(X_test_scaled, y_test)
Run the Jupyter Notebook:

Open Sleep_Efficiency_Prediction_Notebook.ipynb in Jupyter Notebook or Google Colab, follow the steps, and execute each cell in sequence.
Interpretation of Results
Training and Validation Loss: Monitor the loss values during training to ensure the model is learning.
Test Loss (MSE): This value, shown after evaluation, represents the model’s performance in predicting sleep efficiency on unseen data.
Future Improvements
Hyperparameter Tuning: Experiment with different learning rates, batch sizes, and neural network architectures.
Feature Engineering: Explore additional feature engineering techniques to potentially improve model accuracy.
Cross-Validation: Implement k-fold cross-validation to better estimate model performance.
