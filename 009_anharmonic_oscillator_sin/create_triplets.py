from data_utils import triplet_data
import numpy as np
import os

np.random.seed(seed=2)
num_datapoints = 50000

# Create training data
x_train = triplet_data(num_datapoints)

# Create validation data
x_val = triplet_data(num_datapoints//10)

# Create test data
x_test = triplet_data(num_datapoints//5)

# Make directory if it doesn't exist
os.makedirs('data/', exist_ok=True)

# Save data to npy file
np.save('data/x_train.npy', x_train)
np.save('data/x_val.npy', x_val)
np.save('data/x_test.npy', x_test)