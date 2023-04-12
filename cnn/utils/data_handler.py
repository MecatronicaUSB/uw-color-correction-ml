import numpy as np
import matplotlib.pyplot as plt


class DataHandler:
    def __init__(self, train_loader, valid_loader):
        self.train = train_loader is not None
        self.valid = valid_loader is not None

        self.train_loss = np.array([])
        self.valid_loss = np.array([])

        self.acc_train_loss = np.array([])
        self.acc_valid_loss = np.array([])

        self.i = 0

    def epoch_end(self, epoch, lr):
        self.calculate_mean_data()
        self.reset_data()

        print("\nEpoch {0} | Learning rate: {1:.8f}".format(epoch, lr))
        if self.train:
            print("Training   | Cost: {0:.4f}".format(self.acc_train_loss[-1]))

        if self.valid:
            print("Validation | Cost: {0:.4f}".format(self.acc_valid_loss[-1]))

    def calculate_mean_data(self):
        if self.train:
            train_loss = np.mean(self.train_loss)
            self.acc_train_loss = np.append(self.acc_train_loss, train_loss)

        if self.valid:
            valid_loss = np.mean(self.valid_loss)
            self.acc_valid_loss = np.append(self.acc_valid_loss, valid_loss)

    def append_train_loss(self, train_loss):
        self.train_loss = np.append(self.train_loss, train_loss)

    def append_valid_loss(self, valid_loss):
        self.valid_loss = np.append(self.valid_loss, valid_loss)

    def reset_data(self):
        self.reset_losses()

    def reset_losses(self):
        self.train_loss = np.array([])
        self.valid_loss = np.array([])

    def plot(self, loss):
        if loss:
            self.plot_loss(False)
        self.i = 0
        plt.show()

    def plot_loss(self, show):
        if self.train:
            self.figure(
                self.acc_train_loss, "Training loss", "Epochs", "Train loss", False
            )

        if self.valid:
            self.figure(
                self.acc_valid_loss, "Validation loss", "Epochs", "Valid loss", False
            )

        if show:
            plt.show()

    def save_data(self, path):
        # ---------- Creating X axis data
        x = np.arange(0, len(self.acc_train_loss))
        print(self.acc_train_loss)

        # ---------- Assing labels, title and legend
        plt.figure(self.i)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("UNET train and validation loss")
        plt.legend(loc="upper right")

        # ---------- Plotting the loss
        if self.train:
            plt.plot(x, self.acc_train_loss, color="r", label="Train loss")

        if self.valid:
            plt.plot(x, self.acc_valid_loss, color="g", label="Validation loss")

        # ---------- Saving the loss chart
        plt.savefig(path + str(len(self.acc_train_loss)) + "-loss.png")
        plt.clf()
        self.i += 1

    def figure(self, data, title, xlabel, ylabel, increase_i=True):
        if increase_i:
            self.i += 1
        plt.figure(self.i)
        plt.plot(data)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
