import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import statsmodels.api as sm
from sklearn.metrics import classification_report

# ================================
# Step 1: Load and Preprocess Data
# ================================
# Load the dataset
file_path = "CNFTW data csv.csv"  # Update this with your file path
data = pd.read_csv(file_path)
data = data.iloc[::-1].reset_index(drop=True)

testData = pd.read_csv("LINK Data csv.csv")
testData = testData.iloc[::-1].reset_index(drop=True)

# Sort data by timestamp to ensure proper sequential order
data["Timestamp"] = pd.to_datetime(data["DATE8"], errors="coerce")
data = data.sort_values(by="Timestamp")

testData["Timestamp"] = pd.to_datetime(testData["DATE8"], errors="coerce")
testData = testData.sort_values(by="Timestamp")

# Encode "Type" (Buy = 1, Sell = 0)
data["Type"] = data["Type"].map({"Buy": 1, "Sell": 0})
testData["Type"] = testData["Type"].map({"Buy": 1, "Sell": 0})

for col in ["USD Amount", "Token Amount", "ETH amount", "USD Price"]:
    data[col] = data[col].astype(str).str.replace(",", "").str.replace("$", "").replace("-", "0")
    data[col] = pd.to_numeric(data[col], errors="coerce")  # Convert to numeric, invalid entries become NaN
    testData[col] = testData[col].astype(str).str.replace(",", "").str.replace("$", "").replace("-", "0")
    testData[col] = pd.to_numeric(testData[col], errors="coerce")  # Convert to numeric, invalid entries become NaN

data = data.dropna(subset=["USD Amount", "Token Amount", "ETH amount", "USD Price"])
testData = testData.dropna(subset=["USD Amount", "Token Amount", "ETH amount", "USD Price"])

testPred = testData.iloc[10:]

# Normalize numeric fields
scaler = MinMaxScaler()
numeric_columns = ["USD Amount", "Token Amount", "ETH amount", "USD Price"]
data[numeric_columns] = scaler.fit_transform(data[numeric_columns])
testData[numeric_columns] = scaler.fit_transform(testData[numeric_columns])

# Drop rows with NaN in any of the relevant columns
relevant_columns = ["Type"] + numeric_columns
data = data.dropna(subset=relevant_columns)

buy_count = data["Type"].sum()  # Since Buy = 1, sum gives the count of buys
sell_count = len(data["Type"]) - buy_count  # Remaining are sells

print(f"Number of Buys: {buy_count}")
print(f"Number of Sells: {sell_count}")

# ================================
# Step 2: Create Sequences
# ================================
sequence_length = 10  # Use 10 previous transactions to predict the next
features = ["Type"] + numeric_columns
sequences = []
targets = []
testSeq = []
testTar = []

for i in range(len(data) - sequence_length):
    sequences.append(data[features].iloc[i:i+sequence_length].values)
    targets.append(data["Type"].iloc[i + sequence_length])  # Predict next type

for i in range(len(testData) - sequence_length):
    testSeq.append(testData[features].iloc[i:i+sequence_length].values)
    testTar.append(testData["Type"].iloc[i + sequence_length])

    
# Convert to NumPy arrays
X = np.array(sequences)
y = np.array(targets)


# Determine split index based on 80-20 split
split_index = int(0.8 * len(X))  # Use the first 80% for training

# Split the data chronologically
#X_train, X_test = X[:split_index], X[split_index:]
#y_train, y_test = y[:split_index], y[split_index:]

X_train = X
y_train = y
X_test = np.array(testSeq)
y_test = np.array(testTar)

print(f"Training size: {len(X_train)}, Testing size: {len(X_test)}")

# ================================
# Step 3: Build the Model
# ================================

# 
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.2),
    Dense(16, activation="relu"),
    Dense(1, activation="sigmoid")  # Output for binary classification
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# ================================
# Step 4: Train the Model
# ================================
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=10,
    batch_size=32,
    verbose=2
)

# ================================
# Step 5: Evaluate the Model
# ================================
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")

len(y_test)
len(X_test)
# ================================
# Step 6: Save the Model
# ================================
model.save("transaction_predictor.h5")
print("Model saved as transaction_predictor.h5")




#========================
# Correlations and Probabilities
#================

# Predict probability for the test sample
probabilities = model.predict(X_test)  # Predict probabilities for the first 5 samples
for i, prob in enumerate(probabilities):
    print(f"Sample {i + 1}: Probability of Buy = {prob[0]:.4f}")


testData
x = probabilities
x = sm.add_constant(x)
y = testData['USD Amount'].iloc[10:]*testData["Type"].iloc[10:].apply(lambda x: 1 if x == 1 else -1)
modelO = sm.OLS(y,x)
results = modelO.fit()

print(results.summary())



x = probabilities
x = sm.add_constant(x)
y = testData["Type"].iloc[10:]
modelO = sm.OLS(y,x)
results = modelO.fit()

print(results.summary())

    

predictions = (probabilities > 0.5).astype(int)
print(classification_report(y_test, predictions))

testPred['Proba'] = probabilities


#testPred.to_csv('testPred.csv')
