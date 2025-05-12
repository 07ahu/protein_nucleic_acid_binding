from sklearn.preprocessing import StandardScaler
import math
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import numpy as np


# Load the original dataset
dataset = pd.read_excel("/Users/hualj/Desktop/mpdockq/testing_thresholds/Data_spreadsheet.xlsx")

# Select relevant columns for features
mpdockq = dataset[['mpdockq']]
iptm = dataset[['iptm']]
plddt = dataset[['average_plddt']]
contact_pairs = dataset[['contact_pairs']]
average_pae = dataset[['average_pae']]

def fit_with_boxcox(data, name):
    # Extract the first column as a 1D numpy array
    data_values = data.iloc[:, 0].values

    # Ensure the data is positive for Box-Cox transformation
    if (data_values <= 0).any():
        # Shift data to make all values positive by adding a constant
        shift_value = abs(data_values.min()) + 1  # Add enough to make the minimum value positive
        data_values = data_values + shift_value
        print(f"Shifting data for {name} by {shift_value} to make all values positive.")

    # Apply Box-Cox transformation
    transformed_data, best_lambda = stats.boxcox(data_values)
    print(f"Best lambda for {name}: {best_lambda}")

    # Fit the Gaussian (Normal) distribution on the transformed data
    mu, std = stats.norm.fit(transformed_data)

    # Plot histogram of transformed data
    plt.hist(transformed_data, bins=5, density=True, alpha=0.6, color='b')

    # Plot the Gaussian distribution with the estimated parameters
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, mu, std)
    plt.plot(x, p, 'k', linewidth=2)

    plt.title(f'Box-Cox Transformed {name} Gaussian Distribution: Mean = {mu:.2f}, Std = {std:.2f}')
    plt.show()

        # Standardize the transformed data (Z-score normalization)
    scaler = StandardScaler()
    transformed_data_standardized = scaler.fit_transform(transformed_data.reshape(-1, 1))  # reshape for 2D array

    return transformed_data_standardized, best_lambda


def log_transform_iptm(data):
    new_data = []
    for i in data.values:
        new_data.append(math.log10(1.95-i))
    plt.figure(figsize=(10, 8))
    plt.hist(new_data, bins=10, edgecolor='k', alpha=0.7)
    plt.title('iptm scaled')
    plt.grid(True)
    plt.show()
    return new_data

def minmax(data, title):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    # Histogram of Predicted KD Values
    plt.figure(figsize=(10, 8))
    plt.hist(scaled_data, bins=10, edgecolor='k', alpha=0.7)
    plt.title('MINMAX SCALING '+title)
    plt.grid(True)
    plt.show()
    return scaled_data, scaler


iptm_transformed_data = np.array(log_transform_iptm(iptm))
iptm_transformed_data = iptm_transformed_data.reshape(-1, 1)
# Apply the function to each feature column
plddt_transformed_data, plddt_lambda = fit_with_boxcox(plddt, 'plddt')
contact_transformed_data, contact_lambda = fit_with_boxcox(contact_pairs, 'contact')
pae_transformed_data, pae_lambda = fit_with_boxcox(average_pae, 'pae')

iptm_final, iptm_scaler = minmax(iptm_transformed_data,'iptm')
plddt_final, plddt_scaler = minmax(plddt,'plddt')
contact_final, contact_scaler = minmax(contact_transformed_data,'contact')
pae_final, pae_scaler = minmax(pae_transformed_data,'pae')
mpdockq_final, mpdockq_scaler = minmax(mpdockq,'mpdockq')

new_dataframe = {
        'iptm': iptm_final.flatten(),
        'plddt': plddt_final.flatten(),
        'contact_pairs': contact_final.flatten(),
        'pae': pae_final.flatten(),
        'mpodckq': mpdockq_final.flatten()
}
new_dataframe = pd.DataFrame(new_dataframe)
