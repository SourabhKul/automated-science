import os
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
import xgboost as xgb

# Load the .pkl files
file_names = [
    "saved_params.pkl",
    "saved_params2.pkl",
    "saved_params3.pkl",
]

X = []
y = []

# Load the data from all files
for file_name in file_names:
    with open(file_name, "rb") as f:
        data = pickle.load(f)
        for key in dict(data).keys():
            X.append(data[key]["params"])
            y.append(data[key]["distances"])

# Combine data from all files
X = np.vstack(X)
y = np.concatenate(y)
print(X.shape, y.shape)
exit()
max_examples = 10000

# Split the data into train and test sets (e.g., 80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# # Define the neural network model
# model = Sequential()

# # Input layer + 1st hidden layer with 48 neurons, L2 regularization, and Dropout
# model.add(Dense(32, input_dim=15, activation="relu", kernel_regularizer=l2(0.001)))
# model.add(Dropout(0.1))  # Dropout with 30% rate

# # 2nd hidden layer with 48 neurons, L2 regularization, and Dropout
# model.add(Dense(48, activation="relu", kernel_regularizer=l2(0.001)))
# model.add(Dropout(0.1))

# # 3rd hidden layer with 48 neurons, L2 regularization, and Dropout
# model.add(Dense(32, activation="relu", kernel_regularizer=l2(0.001)))
# model.add(Dropout(0.1))

# # Output layer for regression (no activation function)
# model.add(Dense(1))

# # Compile the model for regression using MAPE as the loss function
# model.compile(
#     optimizer=Adam(learning_rate=0.001),
#     loss="mape",
#     metrics=["mape"],
# )

# # Train the model
# history = model.fit(
#     X_train, y_train, epochs=100, batch_size=1024, validation_split=0.2, verbose=1
# )

# # Evaluate the model on the test set
# test_loss, test_mae = model.evaluate(X_test, y_test)
# print(f"Test MSE Loss: {test_loss}, Test MAPE: {test_mae}")

# # Save the model to the same directory
# model.save("trained_model_with_mape_loss.h5")

# print("Model saved as 'trained_model_with_mape_loss.h5'")

# Define the XGBoost model
model = xgb.XGBRegressor(
    objective="reg:squarederror",  # For regression tasks
    n_estimators=200,  # Number of boosting rounds
    learning_rate=0.01,  # Step size shrinkage
    max_depth=20,  # Maximum depth of a tree
    alpha=0.1,  # L1 regularization term
    # num_parallel_tree=10,
    # tree_method="hist",
    # device="cuda",
)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mape = mean_absolute_percentage_error(y_test, y_pred)
print(f"Test MAPE: {mape * 100}%")

# Save the model
model.save_model("trained_model_with_xgboost.json")

print("Model saved as 'trained_model_with_xgboost.json'")
