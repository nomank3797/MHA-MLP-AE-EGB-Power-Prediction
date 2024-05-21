import numpy as np
import pandas as pd
import data_processing  # Custom module for data processing
import mha_mlp_ae_egb  # Custom module for model training and testing

# Load the dataset from the 'raw data' directory
# Assuming 'AASSP.csv' contains the dataset
dataset = pd.read_csv('raw data/AASSP.csv', header=0, low_memory=False,
                      infer_datetime_format=True, parse_dates=[0], index_col=['datetime'])

# Choose data frequency
frequency = 'H'  # Data frequency is set to hourly

# Number of epochs for training
epochs = 100

# Define the filename for the CSV file to save predictions
file_name = 'hourly_actual_predicted_values.csv'

# Clean the data
cleaned_data = data_processing.clean_data(dataset, frequency)

# Split the dataset into input sequences (X) and output sequences (y)
X, y = data_processing.split_sequences(cleaned_data)

# Model training and testing
mha_mlp_ae_egb.mha_mlp_ae_egb_model(X, y, epochs, file_name)
