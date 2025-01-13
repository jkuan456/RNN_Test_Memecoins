# RNN_Test_Memecoins

# Transaction Prediction Model

This repository provides a proof of concept for training and evaluating a binary classification model to predict transaction types (Buy/Sell) using time-series data. The model uses historical transaction data to predict the probability of future transactions being "Buy" or "Sell."

## Features
- Preprocesses transaction data for numeric consistency and sequential order.
- Builds a Long Short-Term Memory (LSTM) neural network for time-series classification.
- Evaluates model performance using metrics such as accuracy and correlation analysis.
- Saves predictions and the trained model for future use.

## Dataset Requirements

The model works with transaction datasets provided as CSV files. The repository includes several sample files with memecoin transactions:
- `CNFTW data csv.csv`
- `LINK Data csv.csv`
- `IMF Data csv.csv`
- `snowbunny data csv.csv`

Each file contain the following columns:

| Column Name          | Description                                                                 |
|----------------------|-----------------------------------------------------------------------------|
| `Time since transaction` | Human-readable time since the transaction occurred (e.g., "13s ago").  |
| `Type`               | Transaction type: "Buy" or "Sell".                                         |
| `USD Amount`         | The transaction value in USD.                                              |
| `Token Amount`       | The quantity of tokens transacted.                                         |
| `ETH amount`         | The equivalent value of the transaction in ETH.                            |
| `USD Price`          | The price of the token in USD.                                             |
| `DATE_URL`           | A URL linking to detailed transaction information.                        |
| `DATE6`              | A truncated identifier related to the transaction.                        |
| `DATE_URL7`          | Another URL linking to transaction details.                               |
| `DATE8`              | Timestamp of the transaction (used for chronological sorting).            |

### Example Data

| Time since transaction | Type | USD Amount | Token Amount | ETH amount | USD Price | DATE_URL                              | DATE6   | DATE_URL7                              | DATE8                     |
|-------------------------|------|------------|--------------|------------|-----------|---------------------------------------|---------|----------------------------------------|---------------------------|
| 13s ago                | Sell | 9.16       | 37,807       | 0.0464     | $0.00     | https://solscan.io/account/CkUZV...  | 4M9fCD  | https://solscan.io/tx/jG5m6i7...       | Dec 26 10:46:17 AM        |
| 14s ago                | Sell | 9.15       | 37,720       | 0.04632    | $0.00     | https://solscan.io/account/CkUZV...  | 4M9fCD  | https://solscan.io/tx/2t9ZJq...        | Dec 26 10:46:16 AM        |
| 15s ago                | Sell | 9.19       | 37,894       | 0.04656    | $0.00     | https://solscan.io/account/CkUZV...  | 4M9fCD  | https://solscan.io/tx/63F8o8...        | Dec 26 10:46:15 AM        |

### Notes:
- The `Type` column must be preprocessed to encode "Buy" as `1` and "Sell" as `0`.
- Numeric columns (`USD Amount`, `Token Amount`, `ETH amount`, and `USD Price`) must be cleaned to remove special characters and converted to numeric types for proper model training.
- The `DATE8` column is crucial for sorting data chronologically.



## Pipeline Overview

### 1. Data Preprocessing
- Cleans and normalizes numeric columns for consistency.
- Encodes transaction types: "Buy" as `1` and "Sell" as `0`.
- Sorts data by timestamp to ensure proper sequential order.
- Creates sequences of 10 transactions to serve as input for the LSTM model.

### 2. Model Architecture
- **Input Layer**: Accepts sequences of 10 transactions with multiple features.
- **LSTM Layers**: Extracts temporal dependencies from the data.
- **Dropout Layers**: Prevents overfitting during training.
- **Dense Layers**: Produces a binary classification output (`Buy` or `Sell`).

### 3. Model Training
- Splits data into training and testing sets.
- Trains using binary cross-entropy loss and the Adam optimizer.
- Validates performance with an 80-20 training-validation split.

### 4. Evaluation
- Evaluates model accuracy on the test dataset.
- Generates predicted probabilities for transaction types.
- Analyzes correlations between predictions and transaction metrics.

### 5. Export
- Saves the trained model as `transaction_predictor.h5`.
- Outputs predictions and probabilities for further analysis.

