import pandas as pd
import numpy as np
import tensorflow as tf
from model import create_pinn_model  # Assuming create_pinn_model is in model.py

# Load normalized soil parameters from Excel file
data = pd.read_excel("ex1.xlsx")
soil_data = data[['cohesion', 'angle_internal_friction', 'unit_weight']]
normalized_data = (soil_data - soil_data.mean()) / soil_data.std()

# Define input data for predictions
X = normalized_data.values

# Load the trained PINN model with weights
pinn_model = create_pinn_model(input_dim=X.shape[1])
pinn_model.load_weights("RNN_WEIGHTS.h5")  # Ensure this path matches the trained model's weights

# Vary soil parameters and predict fatigue life
# Here we vary each parameter over a specified range for analysis
cohesion_range = np.linspace(min(X[:, 0]), max(X[:, 0]), 10)   # Example range for cohesion
friction_angle_range = np.linspace(min(X[:, 1]), max(X[:, 1]), 10)  # Range for angle of internal friction
unit_weight_range = np.linspace(min(X[:, 2]), max(X[:, 2]), 10)     # Range for unit weight

# Prepare to store predictions
results = []

for cohesion in cohesion_range:
    for friction_angle in friction_angle_range:
        for unit_weight in unit_weight_range:
            # Create a single input sample with the varied parameters
            sample_input = np.array([[cohesion, friction_angle, unit_weight]])
            
            # Predict fatigue life using the trained model
            fatigue_life_pred = pinn_model.predict(sample_input)
            
            # Store results with parameters
            results.append({
                "Cohesion": cohesion,
                "Angle of Internal Friction": friction_angle,
                "Unit Weight": unit_weight,
                "Predicted Fatigue Life": fatigue_life_pred.flatten()[0]
            })

# Save the results to a CSV file for analysis
results_df = pd.DataFrame(results)
results_df.to_csv("fatigue_life_predictions.csv", index=False)

print("Fatigue life predictions saved to fatigue_life_predictions.csv")
