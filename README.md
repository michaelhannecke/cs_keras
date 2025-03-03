# Concrete Strength Prediction
This repository contains a series of Jupyter notebooks that demonstrate how to predict concrete strength using neural networks with TensorFlow/Keras. The project explores different model architectures and training approaches to optimize prediction accuracy.
Dataset
The dataset (concrete_data.csv) contains 1030 samples with 8 input features and 1 target variable:

## Features:

- Cement
- Blast Furnace Slag
- Fly Ash
- Water
- Superplasticizer
- Coarse Aggregate
- Fine Aggregate


## Target:

- Compressive Strength

## Notebooks Overview

**a-concrete-strength-prediction.ipynb**

- Basic baseline model with a single hidden layer (10 nodes)
- Uses ReLU activation function
- Adam optimizer and mean squared error loss function
- 30% train-test split using scikit-learn
- Trains for 50 epochs
- Runs 50 iterations with different random splits to evaluate model stability
- Reports mean squared error statistics across all runs

**b-concrete-strength-prediction-with-nomalization.ipynb**

- Extends the baseline model by adding data normalization
- Uses StandardScaler to normalize input features (subtracting mean and dividing by standard deviation)
- Same model architecture (single hidden layer with 10 nodes)
- Demonstrates the importance of feature scaling for neural networks
Detailed comments explaining the normalization process and its benefits
- Reports comprehensive statistics on model performance

**c-concrete-strength-prediction-with-nomalization-100-epochs.ipynb**

- Similar to the previous notebook but increases to 100 iterations
- Uses normalized data
- Single hidden layer architecture with ReLU activation
- Provides an extended evaluation of model stability across more random data splits
- Demonstrates how different data splits affect prediction performance

**d-concrete-strength-prediction-improved-network.ipynb**

- Implements a deeper neural network architecture with three hidden layers
- Each hidden layer has 10 nodes with ReLU activation
- Uses normalized data for better training performance
- Displays detailed model architecture summary
- Evaluates performance over 50 iterations with different random splits
- Calculates additional metrics including RMSE
- Compares the benefits of a deeper network for capturing complex relationships

## Key Findings

Throughout these notebooks, several approaches to improving concrete strength prediction are explored:

- Data normalization significantly improves model training and performance
- Deeper neural networks can capture more complex relationships in the data
- Multiple training iterations with different random splits provide a more robust evaluation of model performance

## Requirements

- Python 3.x
- TensorFlow/Keras
- Pandas
- NumPy
- Scikit-learn

## Usage
Each notebook can be run independently to demonstrate different aspects of the modeling process. The notebooks are designed to be educational and include detailed comments explaining each step of the process.

To run the notebooks:

- Ensure you have all required dependencies installed
- Place the concrete_data.csv file in the data/ directory
- Execute the notebooks in alphabetical order to follow the progression of model improvements