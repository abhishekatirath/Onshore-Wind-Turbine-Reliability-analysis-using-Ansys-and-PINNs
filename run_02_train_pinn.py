import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from model import create_pinn_model  # Assuming create_pinn_model is in model.py

# Load and normalize soil data from Excel file
data = pd.read_excel("B:\PINNs\ex1.xlsx")
soil_data = data[['cohesion', 'angle_internal_friction', 'unit_weight']]
normalized_data = (soil_data - soil_data.mean()) / soil_data.std()

# Define input and output data
X = normalized_data.values
y = data['reliability'].values  # Replace with the actual column name for the output

# Split the data (use only the 70% testing set for this file)
_, X_test, _, y_test = train_test_split(X, y, test_size=0.7, random_state=42)

# Define the shape of the input batch for the testing model
batch_input_shape = (X_test.shape[0], X_test.shape[1])

# Load trained model with weights from 'run_01_train_rnn.py'
pinn_model = create_pinn_model(input_dim=X_test.shape[1])
pinn_model.load_weights("RNN_WEIGHTS.h5")  # Ensure the path matches saved weights

# Evaluate the model on testing data
test_loss, test_mae = pinn_model.evaluate(X_test, y_test, verbose=1)
print(f"Test Loss: {test_loss}")
print(f"Test MAE: {test_mae}")

# Predict and analyze results
y_pred = pinn_model.predict(X_test)

# Save the predictions and actual values for comparison
results_df = pd.DataFrame({"Actual": y_test, "Predicted": y_pred.flatten()})
results_df.to_csv("pinn_test_results.csv", index=False)
