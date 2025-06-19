

# Neural Network Charity Analysis

This project applies a deep learning approach to predict the success of charity funding applications using real-world data. By leveraging TensorFlow and scikit-learn, the goal is to help organizations identify which applications are most likely to be funded successfully.

## Project Overview

- **Data Source:** The dataset contains information about charity applications, such as application type, affiliation, classification, use case, organization type, status, income amount, special considerations, requested amount, and whether the application was successful.
- **Preprocessing:** The data undergoes extensive cleaning:
  - Removal of non-beneficial ID columns (like EIN and NAME)
  - Binning of categorical variables with low frequency (e.g., rare application types and classifications are grouped into "Other")
  - One-hot encoding of categorical features
  - Feature scaling using StandardScaler
- **Model Architecture:**  
  - The model is a simple feedforward neural network (Multi-Layer Perceptron) built with TensorFlow/Keras.
  - The input layer matches the number of engineered features.
  - There is at least one hidden dense layer (with 80 neurons in the example).
  - The output layer uses a single neuron for binary classification (success or failure).
- **Training and Evaluation:**  
  - The data is split into training and testing sets.
  - The model is trained to predict the likelihood of a successful funding application.
  - Performance is evaluated with standard classification metrics.

## Key Technologies

- Python (Jupyter Notebook)
- Pandas, scikit-learn (for data preprocessing and model evaluation)
- TensorFlow/Keras (for neural network construction and training)

## Usage

1. Place your dataset as `charity_data.csv` in the working directory.
2. Run the `AlphabetSoupCharity.ipynb` notebook to preprocess the data, train the model, and review evaluation results.
3. Adjust the neural network architecture or preprocessing steps as needed to experiment and improve performance.

## Purpose

This project demonstrates how neural networks can be used for binary classification problems in the nonprofit sector, providing a framework for predicting the outcomes of charity funding requests.
