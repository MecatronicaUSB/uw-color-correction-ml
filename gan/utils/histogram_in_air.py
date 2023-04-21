import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
import json

HIST_BINS = 100

# ---------- Opening parameters
with open(os.path.dirname(__file__) + "/../parameters.json") as path_file:
    params = json.load(path_file)

output_path = "E:/Descargas/datasets/saved/"
dataset = h5py.File(params["datasets"]["in-air"], "r")

for index in range(672, len(dataset["images"])):
    img = dataset["images"][index]

    img = np.transpose(img, (2, 1, 0)) / 255
    fig = plt.figure(index)
    plt.imshow(img)
    plt.savefig("{0}{1}.jpg".format(output_path, index))
    fig.clear()
    fig.clf()
    plt.clf()

    depth_image = dataset["depths"][index]
    print(
        "{0} | Min: {1:.2f} - Max: {2:.2f} - Range: {3:.2f}".format(
            index,
            np.min(depth_image),
            np.max(depth_image),
            np.max(depth_image) - np.min(depth_image),
        )
    )

# min_values = np.array([])
# max_values = np.array([])

# print("Checking images...")

# for i in range(len(dataset["depths"])):
#     depth_image = np.array(dataset["depths"][i])
#     min_values = np.append(min_values, np.min(depth_image[depth_image != 0]))
#     max_values = np.append(max_values, np.max(depth_image[depth_image != 0]))

# range_values = np.abs(max_values - min_values)

# print(np.where(range_values < 1))

# print("Min depth:")
# print(np.min(min_values))
# print("Max depth:")
# print(np.max(max_values))

# print()

# plt.figure(1)
# plt.title("Min depth histogram")
# plt.hist(min_values, bins=HIST_BINS)

# plt.figure(2)
# plt.title("Max depth histogram")
# plt.hist(max_values, bins=HIST_BINS)

# plt.figure(3)
# plt.title("Range depth histogram")
# plt.hist(range_values, bins=HIST_BINS)

# plt.show()
