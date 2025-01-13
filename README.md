# RNN_Test_Memecoins

Transaction Prediction Model
This repository provides a complete pipeline for training and evaluating a binary classification model to predict transaction types (Buy/Sell) using time-series data. The model uses historical transaction data to predict the probability of future transactions being "Buy" or "Sell."

Features
Preprocesses transaction data for numeric consistency and sequential order.
Builds a Long Short-Term Memory (LSTM) neural network for time-series classification.
Provides detailed model evaluation, including probabilities, correlations, and performance metrics.
Saves the trained model for future use.
Table of Contents
Installation
Dataset Requirements
Pipeline Overview
Usage
Results
Future Enhancements
Installation
To run this project, you need Python 3.8 or above with the following libraries:

bash
Copy code
pip install pandas numpy scikit-learn tensorflow statsmodels
Dataset Requirements
The model expects CSV files with the following columns:

DATE8: Transaction date in a recognized datetime format.
Type: Transaction type, encoded as "Buy" or "Sell."
USD Amount, Token Amount, ETH amount, USD Price: Numeric columns with transaction-related values.
Example data:

DATE8	Type	USD Amount	Token Amount	ETH amount	USD Price
2023-01-01	Buy	100	0.05	0.0025	2000
Pipeline Overview
Data Preprocessing:

Cleans and normalizes numeric data.
Encodes "Buy" as 1 and "Sell" as 0.
Creates sequences of 10 transactions for time-series prediction.
Model Architecture:

LSTM layers with dropout for feature extraction.
Fully connected layers for classification.
Training and Evaluation:

Trains on 80% of the data and evaluates on the remaining 20%.
Computes probabilities and correlation metrics.
Export:

Saves predictions and probabilities for further analysis.
Usage
Clone the repository:

bash
Copy code
git clone https://github.com/your-repo-name/transaction-predictor.git
cd transaction-predictor
Replace CNFTW data csv.csv and LINK Data csv.csv with your dataset files.

Run the pipeline:

bash
Copy code
python transaction_predictor.py
Outputs:

Trained model saved as transaction_predictor.h5.
Classification report and correlation analysis in the console.
Results
Model Accuracy: Achieved through test evaluation.
Probabilities: Predicted likelihood for transactions being "Buy."
Correlation: Assesses the relationship between probabilities and weighted transaction metrics.

