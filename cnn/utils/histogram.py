import matplotlib.pyplot as plt
import numpy as np


def save_rgb_histograms(rgb_tensor, path, title):
    # Convert the tensor to a numpy array and reshape it to a RGB image
    image = np.transpose(rgb_tensor.numpy(), (1, 2, 0)) * 255

    # Extract the three color channels
    r_channel = image[:, :, 0]
    g_channel = image[:, :, 1]
    b_channel = image[:, :, 2]

    # Compute the histograms for each channel
    hist_r, _ = np.histogram(r_channel.flatten(), bins=256, range=(0, 255))
    hist_g, _ = np.histogram(g_channel.flatten(), bins=256, range=(0, 255))
    hist_b, _ = np.histogram(b_channel.flatten(), bins=256, range=(0, 255))

    # Plot the histograms in a single figure
    _, ax = plt.subplots()
    ax.plot(hist_r, color="red", label="Red")
    ax.plot(hist_g, color="green", label="Green")
    ax.plot(hist_b, color="blue", label="Blue")
    ax.set_xlabel("Intensity")
    ax.set_ylabel("Frequency")
    ax.set_title(title)
    ax.legend(loc="upper right")

    plt.savefig(path)
