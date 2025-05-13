# Import necessary libraries
from keras.layers import Dense, Dropout
from keras.regularizers import l2
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from combined import *
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from keras.metrics import MeanAbsoluteError
from keras.losses import MeanSquaredError


FLAGS = flags.FLAGS
flags.DEFINE_string('training_dataset', '.', 'dataset for training')
flags.DEFINE_string('testing_dataset','.','dataset for testing')

# Load Data
data = pd.read_excel(FLAGS.training_datset)

features = new_dataframe  # Features loaded from combined.py
print('features', features)
kd_values = data[['kd']]  # Target values

# Normalize Features
scaler = MinMaxScaler()
X = features
y = scaler.fit_transform(kd_values.values)  # Target (assumes single-column kd)

# Define Model Creation Function
def create_model():
    model = Sequential([
        Dense(32, input_dim=X.shape[1], activation='relu', 
              kernel_regularizer=l2(0.01)),  # L2 regularization
        Dropout(0.5),
        Dense(16, activation='relu', kernel_regularizer=l2(0.01)),
        Dropout(0.5),
        Dense(1, activation='linear')  # Linear activation for regression output
    ])
    model.compile(optimizer=Adam(learning_rate=0.001),     loss=MeanSquaredError(), metrics=[MeanAbsoluteError()])
    return model

# Early Stopping Callback
early_stopping = EarlyStopping(
    monitor='val_loss',  # Monitors validation loss
    patience=10,         # Stops training after 10 epochs of no improvement
    restore_best_weights=True  # Restores the best model weights
)

# Split Data into Training and Validation Sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and Train Model
model = create_model()
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,           # Max number of epochs
    batch_size=16,        # Mini-batch size
    callbacks=[early_stopping],  # Early stopping callback
    verbose=1             # Display training progress
)

# Evaluate Model
train_loss, train_mae = model.evaluate(X_train, y_train, verbose=0)
val_loss, val_mae = model.evaluate(X_val, y_val, verbose=0)
print(f"Training Loss: {train_loss:.4f}, Training MAE: {train_mae:.4f}")
print(f"Validation Loss: {val_loss:.4f}, Validation MAE: {val_mae:.4f}")

model.save('model_sigmoid.h5')


# Plot Training and Validation Loss
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.show()

dataset_path = FLAGS.testing_dataset
new_data = pd.read_excel(dataset_path)

transformed_iptm = log_transform_iptm(new_data[['iptm']])
transformed_plddt = new_data[['average_plddt']] ** plddt_lambda
transformed_contact = new_data[['contact_pairs']] ** contact_lambda
transformed_pae = new_data[['average_pae']] ** pae_lambda

# log iptm
# raise powers to plddt_lambda, contact_lambda, pae_lambda

# then use iptm_scaler, plddt_scaler, contact_scaler, pae_scaler, mpdockq_scaler
final_test_iptm = iptm_scaler.fit_transform(np.array(transformed_iptm).reshape(-1, 1))
final_test_plddt = plddt_scaler.fit_transform(np.array(transformed_plddt).reshape(-1, 1))
final_test_contact = contact_scaler.fit_transform(np.array(transformed_contact).reshape(-1, 1))
final_test_pae = pae_scaler.fit_transform(np.array(transformed_pae).reshape(-1, 1))
final_test_mpdockq = mpdockq_scaler.fit_transform(new_data[['mpdockq']])


df= {
        'iptm': final_test_iptm.flatten(),
        'plddt': final_test_plddt.flatten(),
        'contact_pairs': final_test_contact.flatten(),
        'pae': final_test_pae.flatten(),
        'mpodckq': final_test_mpdockq.flatten()
}
df = pd.DataFrame(df)

# Load the previously saved model
model = load_model('model_sigmoid.h5')
predictions = model.predict(df)

# Assuming 'title' column exists in new_data
titles = new_data['title'].values  # Extract titles as a NumPy array
predictions_original_scale = scaler.inverse_transform(predictions)
predicted_values = predictions_original_scale.flatten()

# Print each title with its corresponding prediction
print("\n=== Predictions with Titles ===")
actual_kd_values = new_data[['kd']].values
for title, actual, predicted in zip(titles, actual_kd_values.flatten(), predictions_original_scale.flatten()):
    print(f"Title: {title}, Actual KD: {actual:.4f}, Predicted KD: {predicted:.4f}")

