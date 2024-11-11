import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Lambda, Concatenate
from tensorflow.keras.optimizers import Adam

# Function to create the PINN model for reliability analysis
def create_pinn_model(input_dim, layers=[64, 64, 64], learning_rate=0.001):
    # Input layer
    inputs = Input(shape=(input_dim,))
    
    # Hidden layers
    x = inputs
    for layer_size in layers:
        x = Dense(layer_size, activation='tanh')(x)

    # Output layer (reliability prediction)
    outputs = Dense(1, activation='linear')(x)

    # Define the model
    model = Model(inputs=inputs, outputs=outputs)

    # Compile the model
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    
    return model

# Example usage
if __name__ == "__main__":
    # Load and normalize data from the Excel sheet (dummy data as placeholder)
    # Assuming that the soil data (cohesion, angle of internal friction, unit weight) are in a file named 'ex1.xlsx'
    import pandas as pd
    data = pd.read_excel("B:\PINNs\ex1.xlsx")

    # Extract and normalize input features
    soil_data = data[['cohesion', 'angle_internal_friction', 'unit_weight']]
    normalized_data = (soil_data - soil_data.mean()) / soil_data.std()  # Standard normalization

    # Define input and output data
    X = normalized_data.values
    y = data['reliability'].values  # Assuming 'reliability' column exists for supervised training

    # Split data into training and testing sets
    split_idx = int(0.3 * len(X))  # 30% training data
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Create and train the PINN model
    pinn_model = create_pinn_model(input_dim=X.shape[1])
    pinn_model.fit(X_train, y_train, epochs=100, batch_size=16, validation_data=(X_test, y_test))

    # Save the model (optional)
    pinn_model.save('pinn_model.h5')
