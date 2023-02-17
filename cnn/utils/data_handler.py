import numpy as np
import matplotlib.pyplot as plt


class DataHandler():
    def __init__(self, train_loader, valid_loader):
        self.train = train_loader is not None
        self.valid = valid_loader is not None

        self.train_loss = np.array([])
        self.valid_loss = np.array([])

        self.train_metric = np.array([])
        self.valid_metric = np.array([])

        self.acc_train_loss = np.array([])
        self.acc_valid_loss = np.array([])

        self.acc_train_metric = np.array([])
        self.acc_valid_metric = np.array([])

        self.i = 0

    def epoch_end(self, epoch, lr):
        self.calculate_mean_data()
        self.reset_data()

        print('\nEpoch {0} | Learning rate: {1:.8f}'.format(epoch, lr))
        if self.train:
            print('Training   | Cost: {0:.4f}'.format(
                self.acc_train_loss[-1], self.acc_train_metric[-1]))
        if self.valid:
            print('Validation | Cost: {0:.4f}'.format(
                self.acc_valid_loss[-1], self.acc_valid_metric[-1]))

    def calculate_mean_data(self):
        if self.train:
            train_loss = np.mean(self.train_loss)
            train_metric = np.mean(self.train_metric)
            self.acc_train_loss = np.append(self.acc_train_loss, train_loss)
            self.acc_train_metric = np.append(
                self.acc_train_metric, train_metric)

        if self.valid:
            valid_loss = np.mean(self.valid_loss)
            valid_metric = np.mean(self.valid_metric)
            self.acc_valid_loss = np.append(self.acc_valid_loss, valid_loss)
            self.acc_valid_metric = np.append(
                self.acc_valid_metric, valid_metric)

    def append_train_loss(self, train_loss):
        self.train_loss = np.append(self.train_loss, train_loss)

    def append_valid_loss(self, valid_loss):
        self.valid_loss = np.append(self.valid_loss, valid_loss)

    def reset_data(self):
        self.reset_losses()
        self.reset_metrics()

    def reset_losses(self):
        self.train_loss = []
        self.valid_loss = []

    def reset_metrics(self):
        self.train_metric = []
        self.valid_metric = []

    def plot(self, loss, metric):
        if loss:
            self.plot_loss(False)
        if metric:
            self.plot_metric(False)
        self.i = 0
        plt.show()

    def plot_loss(self, show):
        if self.train:
            self.figure(self.acc_train_loss, 'Training loss',
                        'Epochs', 'Train loss', False)

        if self.valid:
            self.figure(self.acc_valid_loss, 'Validation loss',
                        'Epochs', 'Valid loss', False)

        if show:
            plt.show()

    def plot_metric(self, show):
        if self.train:
            self.figure(self.acc_train_metric, 'Training metric',
                        'Epochs', 'Train metric')

        if self.valid:
            self.figure(self.acc_valid_metric, 'Validation metric',
                        'Epochs', 'Valid metric')

        if show:
            plt.show()

    def figure(self, data, title, xlabel, ylabel, increase_i=True):
        if increase_i:
            self.i += 1
        plt.figure(self.i)
        plt.plot(data)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
