import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
import json

HIST_BINS = 100

# ---------- Opening parameters
with open(os.path.dirname(__file__) + "/../parameters.json") as path_file:
    params = json.load(path_file)

dataset = h5py.File(params["datasets"]["in-air"], 'r')

min_values = np.array([])
max_values = np.array([])

print('Checking images...')

for i in range(len(dataset['depths'])):
    depth_image = np.array(dataset['depths'][i])
    min_values = np.append(min_values, np.min(depth_image[depth_image != 0]))
    max_values = np.append(max_values, np.max(depth_image[depth_image != 0]))

range_values = np.abs(max_values - min_values)

print('Min depth:')
print(np.min(min_values))
print('Max depth:')
print(np.max(max_values))

print()

plt.figure(1)
plt.title('Min depth histogram')
plt.hist(min_values, bins=HIST_BINS)

plt.figure(2)
plt.title('Max depth histogram')
plt.hist(max_values, bins=HIST_BINS)

plt.figure(3)
plt.title('Range depth histogram')
plt.hist(range_values, bins=HIST_BINS)

plt.show()
