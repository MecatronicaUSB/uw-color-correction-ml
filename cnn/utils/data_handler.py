import numpy as np
import matplotlib.pyplot as plt
import datetime


class DataHandler:
    def __init__(self, train_loader, valid_loader):
        self.train = train_loader is not None
        self.valid = valid_loader is not None

        self.train_loss = np.array([])
        self.valid_loss = np.array([])

        self.acc_train_loss = np.array([])
        self.acc_valid_loss = np.array([])

        self.best_valid_loss = np.inf

        self.i = 0

    def epoch_end(self, epoch, lr):
        self.calculate_mean_data()
        self.reset_data()

        print(
            "Finished at: {0} | Learning rate: {1:.8f}".format(
                datetime.datetime.now(), lr
            )
        )

        if self.train:
            print("Training   | Cost: {0:.4f}".format(self.acc_train_loss[-1]))

        if self.valid:
            print("Validation | Cost: {0:.4f}".format(self.acc_valid_loss[-1]))

            if self.acc_valid_loss[-1] < self.best_valid_loss:
                self.best_valid_loss = self.acc_valid_loss[-1]
                return True

        return None

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

    def save_data(self, path):
        # ---------- Creating X axis data
        x = np.arange(
            0,
            len(self.acc_train_loss)
            if self.train
            else len(self.acc_valid_loss)
            if self.valid
            else 0,
        )

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
