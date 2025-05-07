import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from absl import flags, app, logging

# Step 1: Train and Save the Model

# Load the original dataset

FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', '.', 'dataset for training')

# before used: data = pd.read_excel("/Users/hualj/Desktop/mpdockq/testing_thresholds/Data_spreadsheet.xlsx")
data = pd.read_excel(FLAGS.datset)

# Select relevant columns for features and target
data2 = data[['iptm', 'average_plddt', 'mpdockq', 'contact_pairs', 'average_pae']]  # Features

kd_values = data['kd']  # Target variable, 'kd'
filtered_kd_values = kd_values[kd_values < 200]
filtered_data = np.log(kd_values+1)
# Select the column(s) you want to visualize
column_to_plot = 'kd'  # Replace with the column name you want to plot
bins = 5  # Number of bins for the histogram

#print('average log',sum(filtered_data)/len(filtered_data))
#print('std dev log', np.std(filtered_data))

# Create the histogram

plt.figure(figsize=(10, 6))
plt.hist(filtered_data, bins=bins, color='blue', edgecolor='black', alpha=0.7)
plt.title(f'LOG Distribution of {column_to_plot}')
plt.xlabel(column_to_plot)
plt.ylabel('Frequency')
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.show()
"""
# Show the plot
plt.show()
print('fitl', filtered_kd_values)
new_kd = 1/filtered_kd_values
print('new_kd', new_kd)
# Create the histogram with fewer bins (e.g., 10 bins)
plt.figure(figsize=(10, 6))
plt.hist(new_kd, bins=5, color='blue', edgecolor='black', alpha=0.7)
plt.title('1/KD')
plt.xlabel('KD')
plt.ylabel('Frequency')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Show the plot
plt.show()

"""
#print('1/kd average',sum(new_kd)/len(new_kd))
#print('std dev', np.std(new_kd))

print('average',sum(filtered_data)/len(filtered_data))
print('std dev', np.std(filtered_data))


# Binarize the 'kd' values for classification
kd_threshold = 2.93 # Adjust this threshold based on your needs
labels = (filtered_data < kd_threshold).astype(int)  # Convert to binary labels (0 = bad binding, 1 = good binding)

# Prepare features
features = data2  # Use the selected features

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Build the classification model
model = Sequential([
    Dense(64, activation='relu', input_dim=X_train.shape[1]),
    Dropout(0.5),
    Dense(32, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')  # Output layer for binary classification (2 output units)
])

# Compile the model with sparse_categorical_crossentropy
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

# Save the trained model
model.save('/Users/hualj/Desktop/mpdockq/testing_thresholds/model.h5')

# Step 2: Load the Saved Model and Predict on New Data (without 'kd')

# Load the new dataset (without 'kd' values)
dataset_path = '/Users/hualj/Desktop/mpdockq/testing_thresholds/testing_data.xlsx'
new_data = pd.read_excel(dataset_path)

new_kd_values = np.log(new_data['kd']+1)

# Assume the new dataset contains the same features as the original dataset, but without 'kd'
new_features = new_data[['iptm', 'average_plddt', 'mpdockq', 'contact_pairs', 'average_pae']]

# Load the previously saved model
model = load_model('/Users/hualj/Desktop/mpdockq/testing_thresholds/model.h5')

# Make predictions for the new data
predictions = model.predict(new_features)

# Get the probabilities for "Bad Binding" (class 0) and "Good Binding" (class 1)
predicted_probs = predictions  # The output will contain probabilities for both classes (Good and Bad Binding)

# Combine the predicted probabilities with the titles (from the new dataset)
new_data['Bad Binding Probability'] = predicted_probs[:, 0]
new_data['Good Binding Probability'] = predicted_probs[:, 1]

correct_predictions = 0
total = len(new_kd_values)

# Print out the results, including titles
for index, row in new_data.iterrows():
    print(f"Title: {row['title']}")
    bad_bind_prob = row['Bad Binding Probability']
    good_bind_prob = row['Good Binding Probability']
    print(f"  Bad Binding Probability: {row['Bad Binding Probability']:.4f}")
    print(f"  Good Binding Probability: {row['Good Binding Probability']:.4f}")
    print("-" * 50)

    if bad_bind_prob>good_bind_prob:
        if new_kd_values[index] > kd_threshold:
            correct_predictions+=1
    elif good_bind_prob>bad_bind_prob:
        if new_kd_values[index]<kd_threshold:
            correct_predictions+=1
print('accuracy', correct_predictions/total)


# Optionally, save the predictions with titles to a new file
new_data.to_excel('/Users/hualj/Desktop/mpdockq/testing_thresholds/testing_data.xlsx', index=False)

# Optional: Visualizing Training and Validation Loss & Accuracy
# Training vs Validation Loss
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training vs Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

# Training vs Validation Accuracy
plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training vs Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

combined = new_data['Good Binding Probability'] - new_data['Bad Binding Probability']
# Scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(new_kd_values, combined, color='blue', alpha=0.7, edgecolor='black')

# Titles and labels
plt.title('Combined Probabilities vs KD Values', fontsize=16)
plt.xlabel('KD Values', fontsize=14)
plt.ylabel('Combined Probabilities (Good - Bad Binding)', fontsize=14)

# Grid and show
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()
