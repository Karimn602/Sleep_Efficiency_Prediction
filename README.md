# Sleep Efficiency Prediction using a Feed-Forward Neural Network
This project predicts sleep efficiency based on various factors, including age, sleep duration, REM sleep percentage, and lifestyle habits. The model is implemented as a feed-forward neural network built with TensorFlow and Keras.

# Dataset
The dataset used in this project contains several features related to sleep patterns, demographics, and lifestyle. Each row represents an individual's sleep data, including the following key features:

1. Age: The age of the individual.
2. Gender: Gender of the individual.
3. Sleep Duration: Number of hours spent sleeping.
4. REM Sleep Percentage: Percentage of sleep spent in REM.
5. Deep Sleep Percentage: Percentage of sleep spent in deep sleep.
6. Light Sleep Percentage: Percentage of sleep spent in light sleep.
7. Awakenings: Number of times the individual woke up during the night.
8. Caffeine Consumption: Amount of caffeine consumed.
9. Alcohol Consumption: Amount of alcohol consumed.
10. Smoking Status: Whether the individual smokes.
11. Exercise Frequency: Frequency of exercise.
The target variable is Sleep Efficiency, which measures how efficiently the individual sleeps.

# Dataset Source
You can find various sleep datasets on platforms like Kaggle or Hugging Face Datasets. Alternatively, you can upload a compatible dataset directly for this project.For my test I used a dataset I found in kaggle.

# Model Architecture
The model is a simple feed-forward neural network with the following structure:

1. Input Layer: Receives multiple input features from the dataset.
2. Hidden Layer 1: A dense layer with 64 neurons and ReLU activation.
3. Hidden Layer 2: A dense layer with 32 neurons and ReLU activation.
4. Output Layer: A single neuron for predicting sleep efficiency (as this is a regression task).
5. The loss function used is mean squared error (MSE), and the optimizer used is Adam with a learning rate of 0.001.

# Code Structure and Usage
The project is organized into modules for easy use and reusability:

 1. dataset_module.py
This module contains the load_and_preprocess_data function, which handles data loading, preprocessing, and scaling.

2. model_module.py
This module contains the NeuralNetworkModel class, which defines the neural network structure, training, and evaluation methods.

3. Jupyter Notebook (Sleep_Efficiency_Prediction_Notebook.ipynb)
This notebook demonstrates how to use the modules to load data, initialize the model, train, and evaluate it.

## Instructions for Running the Code
1. Install Dependencies:

Ensure you have installed the necessary libraries (tensorflow, scikit-learn, and pandas).

2. Using the Modules:

Import and use the load_and_preprocess_data function from dataset_module.py to load and preprocess the dataset.
Import and use the NeuralNetworkModel class from model_module.py to initialize the model, train it on your data, and evaluate its performance.
Run the Jupyter Notebook:

3. Open Sleep_Efficiency_Prediction_Notebook.ipynb in Jupyter Notebook or Google Colab, follow the steps, and execute each cell in sequence

# Interpretation of Results
Training and Validation Loss: Monitor the loss values during training to ensure the model is learning.
Test Loss (MSE): This value, shown after evaluation, represents the model’s performance in predicting sleep efficiency on unseen data.
# Future Improvements
1. Hyperparameter Tuning: Experiment with different learning rates, batch sizes, and neural network architectures.
2. Feature Engineering: Explore additional feature engineering techniques to potentially improve model accuracy.
3. Cross-Validation: Implement k-fold cross-validation to better estimate model performance.



