import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, RNN
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from model import create_rnn_model  # Assuming create_rnn_model is in model.py

# Load data from Excel file
data = pd.read_excel("B:\PINNs\ex1.xlsx")

# Extract and normalize soil parameters
soil_data = data[['cohesion', 'angle_internal_friction', 'unit_weight']]
normalized_data = (soil_data - soil_data.mean()) / soil_data.std()

# Define input and output data
X = normalized_data.values
y = data['reliability'].values  # Replace with the appropriate column name for output

# Split data into 30% training and 70% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=42)

# Define the batch shape for RNN
batch_input_shape = (X_train.shape[0], X_train.shape[1])

# Define initial damage value as specified
d0RNN = np.asarray([0.0]) * np.ones((X_train.shape[0], 1))

# Load the pretrained MLP model
mlp_model = tf.keras.models.load_model("MLP_PLANE.h5")  # Adjust the path as necessary
mlp_model.trainable = True

# Set up the RNN model with PINN structure
rnn_model = create_rnn_model(mlp_model, d0RNN, batch_input_shape, lowBounds_delgrs=-1, upBounds_delgrs=1, myDtype='float32')

# Compile the model
rnn_model.compile(optimizer=Adam(learning_rate=1e-3), loss='mse', metrics=['mae'])

# Training parameters
EPOCHS = 100
BATCH_SIZE = 16

# Train the model
history = rnn_model.fit(
    X_train, y_train, 
    epochs=EPOCHS, 
    batch_size=BATCH_SIZE, 
    validation_data=(X_test, y_test),
    verbose=1
)

# Save the model weights after training
rnn_model.save_weights("RNN_WEIGHTS.h5")

# Save the training history (loss values)
history_df = pd.DataFrame(history.history)
history_df.to_csv("training_history.csv", index=False)
