import numpy as np
import matplotlib.pyplot as plt
import datetime


class DataHandler:
    def __init__(self, validation):
        self.valid = validation is not None

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

        print("Training   | Cost: {0:.4f}".format(self.acc_train_loss[-1]))

        if self.valid:
            print("Validation | Cost: {0:.4f}".format(self.acc_valid_loss[-1]))

            if self.acc_valid_loss[-1] < self.best_valid_loss:
                self.best_valid_loss = self.acc_valid_loss[-1]
                return True

        return None

    def calculate_mean_data(self):
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
        # ---------- Saving the loss to txt files
        np.savetxt(
            path + "train-loss.txt",
            self.acc_train_loss,
            delimiter=",",
        )

        if self.valid:
            np.savetxt(
                path + "valid-loss.txt",
                self.acc_valid_loss,
                delimiter=",",
            )

        # ---------- Saving the loss to jpg files
        self.save_figure(
            self.acc_train_loss,
            self.acc_valid_loss,
            path + "loss.jpg",
        )

    def save_data_from_epoch(self, from_epoch, path):
        # ---------- Simplifying the data from the given epoch
        simplified_train_loss = self.acc_train_loss[from_epoch:]
        simplified_valid_loss = self.acc_valid_loss[from_epoch:]

        # ---------- Saving the loss to jpg files
        self.save_figure(
            simplified_train_loss,
            simplified_valid_loss,
            path + "loss-from-epoch.jpg",
        )

    def save_figure(self, train_data, valid_data, path):
        # ---------- Creating X axis data
        x = np.arange(0, len(train_data))

        # ---------- Patch for the plot
        if self.valid:
            valid_data[0] = train_data[0] * 1.1

        # ---------- Plotting the loss
        plt.figure(self.i)
        plt.plot(x, train_data, color="r", label="Train loss")

        if self.valid:
            plt.plot(x, valid_data, color="g", label="Validation loss")

        # ---------- Assing labels, title and legend
        self.set_plt_data()
        plt.ylim(
            0,
            max(np.max(train_data), np.max(valid_data) if self.valid else 0) * 1.15,
        )

        # ---------- Saving the loss chart
        plt.savefig(path)
        plt.clf()

        self.i += 1

    def set_plt_data(self):
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("UNET train and validation loss")
        plt.legend(loc="upper right")
