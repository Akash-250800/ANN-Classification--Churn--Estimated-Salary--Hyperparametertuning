# ANN Churn Classification with Estimated Salary Prediction and Hyper-Parameter Tuning

This project demonstrates the use of an Artificial Neural Network (ANN) to classify customer churn, predict estimated salary, and optimize model performance using hyper-parameter tuning.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Hyper-Parameter Tuning](#hyper-parameter-tuning)
- [Installation](#installation)
- [Usage](#usage)
- [Evaluation](#evaluation)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

This project aims to solve two tasks:
1. **Customer Churn Classification**: Predict whether a customer is likely to churn (classification task).
2. **Estimated Salary Prediction**: Predict a customer's estimated salary (regression task).
  
Additionally, hyper-parameter tuning is applied to optimize the ANN’s performance for both tasks.

## Dataset

The dataset consists of the following features:
- **Demographic Information**: Age, Gender, Geography
- **Account Information**: Balance, Tenure, Number of Products, Estimated Salary
- **Customer Status**: Whether the customer churned (1: Yes, 0: No)

## Model Architecture

The model consists of two parts:
1. **Churn Classification Branch**:
   - **Input Layer**: Takes demographic and account information.
   - **Hidden Layers**: Two fully connected layers with ReLU activation.
   - **Output Layer**: Sigmoid activation for binary classification (churn or not).
   - **Loss Function**: Binary Crossentropy.

2. **Estimated Salary Regression Branch**:
   - **Input Layer**: Same input features as the churn classification branch.
   - **Hidden Layers**: Two fully connected layers with ReLU activation.
   - **Output Layer**: A single neuron for salary prediction.
   - **Loss Function**: Mean Squared Error (MSE).

The model shares some layers for both tasks but branches out after a certain depth for task-specific output.

## Hyper-Parameter Tuning

Hyper-parameter tuning is used to find the best combination of parameters such as:
- **Number of hidden layers**
- **Number of neurons in each layer**
- **Learning rate**
- **Batch size**
- **Epochs**

Tuning is performed using techniques like Grid Search or Random Search, optimizing both the classification and regression objectives.

## Installation

To run the project locally, clone the repository and install the dependencies:

```bash
git clone https://github.com/yourusername/ann-churn-salary-hyperparameter-tuning.git
cd ann-churn-salary-hyperparameter-tuning
pip install -r requirements.txt


Evaluation
Churn Classification Metrics:
Accuracy
Precision

Estimated Salary Regression Metrics:
Mean Absolute Error (MAE)
Mean Squared Error (MSE)

Results
After hyper-parameter tuning, the model achieved the following results:

Churn Classification:

Accuracy: XX%
Precision: XX%

Estimated Salary Prediction:

MAE: XX
MSE: XX

Contributing
Contributions are welcome! If you have ideas for improvement, feel free to submit a pull request or open an issue.
License
This project is licensed under the MIT License. See the LICENSE file for more details.


This version adds hyper-parameter tuning, explaining the approach and how it integrates into the project. Let me know if you'd like any modifications!
"# Akash-250800-ANN-Classification--Churn--Estimated-Salary--Hyperparametertuning" 
