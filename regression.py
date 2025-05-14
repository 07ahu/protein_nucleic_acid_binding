import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from absl import flags, app, logging

# Load the data
flags.DEFINE_string('training_dataset', '.', 'dataset for training')
data = pd.read_excel(FLAGS.training_datset)
data2 = data[['iptm', 'average_plddt', 'mpdockq', 'contact_pairs', 'average_pae', 'kd']]

# Handle outliers in KD values (clipping extreme values)
data2['kd_clipped'] = data2['kd'].clip(upper=50000)

# Apply log transformation to KD values
data2['kd_log'] = np.log1p(data2['kd_clipped'])  # log1p ensures non-negative KD values

# Prepare features and normalize using MinMaxScaler
features = data2[['iptm', 'average_plddt', 'mpdockq', 'contact_pairs', 'average_pae']].values
scaler_features = MinMaxScaler()
features_scaled = scaler_features.fit_transform(features)

# Use log-transformed KD as labels
labels = data2['kd_log'].values

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features_scaled, labels, test_size=0.2, random_state=42)

# Build the regression model
model = Sequential([
    Dense(128, activation='relu', input_dim=X_train.shape[1], kernel_regularizer=l2(0.001)),
    Dropout(0.3),
    Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
    Dropout(0.3),
    Dense(32, activation='relu', kernel_regularizer=l2(0.001)),
    Dropout(0.3),
    Dense(1, activation='relu')  # Ensures non-negative KD predictions
])

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='mse',  # Mean Squared Error
              metrics=['mae'])  # Mean Absolute Error for tracking

# Set up early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
history = model.fit(X_train, y_train, 
                    epochs=100, 
                    batch_size=32, 
                    validation_data=(X_test, y_test), 
                    callbacks=[early_stopping])

# Evaluate the model
loss, mae = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}, Test MAE: {mae}")

# Make predictions
predictions_log = model.predict(X_test).flatten()

# Inverse log-transform predictions and true values to the original scale
predictions_original = (np.expm1(predictions_log))
y_test_original = np.expm1(y_test)

# Evaluate predictions on the original scale
mae_original = mean_absolute_error(y_test_original, predictions_original)
mse_original = mean_squared_error(y_test_original, predictions_original)
r2 = r2_score(y_test_original, predictions_original)

print(f"Mean Absolute Error (MAE): {mae_original}")
print(f"Mean Squared Error (MSE): {mse_original}")
print(f"R-squared (R²): {r2}")

# Plot Actual vs. Predicted
plt.figure(figsize=(10, 8))
plt.scatter(y_test_original, predictions_original, alpha=0.7, color='blue', edgecolor='k')
plt.plot([y_test_original.min(), y_test_original.max()],
         [y_test_original.min(), y_test_original.max()], 
         'r--', lw=2)  # Line representing perfect predictions
plt.xlabel("Actual KD (Binding Value)")
plt.ylabel("Predicted KD Value")
plt.title("Actual vs. Predicted KD Value")
plt.grid(True)
plt.show()

# Calculate residuals
residuals = y_test_original - predictions_original

# Plot Residuals
plt.figure(figsize=(10, 8))
plt.scatter(predictions_original, residuals, alpha=0.7, color='purple', edgecolor='k')
plt.axhline(0, color='red', linestyle='--', lw=2)
plt.xlabel("Predicted KD Value")
plt.ylabel("Residuals (True - Predicted)")
plt.title("Residuals Plot")
plt.grid(True)
plt.show()

# Plot the training and validation loss over epochs
plt.figure(figsize=(10, 8))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.show()

# Histogram of Residuals
plt.figure(figsize=(10, 8))
plt.hist(residuals, bins=30, edgecolor='k', alpha=0.7)
plt.title('Histogram of Residuals')
plt.xlabel('Residuals (True - Predicted)')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# Histogram of Predicted KD Values
plt.figure(figsize=(10, 8))
plt.hist(predictions_original, bins=30, edgecolor='k', alpha=0.7)
plt.title('Histogram of Predicted KD Values')
plt.xlabel('Predicted KD Value')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# Plot 1/Actual KD vs Predicted KD
plt.figure(figsize=(10, 8))
plt.scatter(1 / y_test_original, predictions_original, alpha=0.7, color='blue', edgecolor='k')
plt.plot([1 / y_test_original.min(), 1 / y_test_original.max()],
         [predictions_original.min(), predictions_original.max()],
         'r--', lw=2)  # Line representing the trend
plt.xlabel("1 / Actual KD (Binding Value)")
plt.ylabel("Predicted KD Value")
plt.title("1 / Actual KD vs. Predicted KD Value")
plt.grid(True)
plt.show()
