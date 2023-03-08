import numpy as np
import matplotlib.pyplot as plt


class DataHandler():
    def __init__(self, name, train_loader=True, valid_loader=True):
        self.name = name
        self.train = train_loader is not None
        self.valid = valid_loader is not None

        self.train_loss = np.array([])
        self.valid_loss = np.array([])

        self.acc_train_loss = np.array([])
        self.acc_valid_loss = np.array([])

        self.best_valid_loss = float("inf")

        self.i = 0

    def epoch_end(self, lr):
        self.calculate_mean_data()
        self.reset_data()

        print('\n{0} | Learning rate: {1:.8f}'.format(self.name, lr))
        if self.train:
            print('Training   | Cost: {0:.4f}'.format(self.acc_train_loss[-1]))
        if self.valid:
            print('Validation | Cost: {0:.4f}'.format(self.acc_valid_loss[-1]))

    def calculate_mean_data(self):
        if self.train:
            train_loss = np.mean(self.train_loss)
            self.acc_train_loss = np.append(self.acc_train_loss, train_loss)

        if self.valid:
            valid_loss = np.mean(self.valid_loss)
            self.acc_valid_loss = np.append(self.acc_valid_loss, valid_loss)

            if valid_loss < self.best_valid_loss:
                self.best_valid_loss = valid_loss

    def append_train_loss(self, train_loss):
        self.train_loss = np.append(self.train_loss, train_loss)

    def append_valid_loss(self, valid_loss):
        self.valid_loss = np.append(self.valid_loss, valid_loss)

    def reset_data(self):
        self.reset_losses()

    def reset_losses(self):
        self.train_loss = []
        self.valid_loss = []

    def plot(self, loss):
        if loss:
            self.plot_loss(False)
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

    def figure(self, data, title, xlabel, ylabel, increase_i=True):
        if increase_i:
            self.i += 1
        plt.figure(self.i)
        plt.plot(data)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

    def custom_multiple_epoch_end(self, epoch, discriminator_data_handler):
        self.calculate_mean_data()
        self.reset_data()

        discriminator_data_handler.calculate_mean_data()
        discriminator_data_handler.reset_data()

        print('\n-------------- Epoch: {0} --------------'.format(epoch))
        # print('Training:')
        print('Generator cost: {0:.4f}'.format(self.acc_train_loss[-1]))
        print('Discriminator cost: {0:.4f}'.format(
            discriminator_data_handler.acc_train_loss[-1]))

        # print('\nValidation:')
        # print('Generator cost: {0:.4f}'.format(self.acc_valid_loss[-1]))
        # print('Discriminator cost: {0:.4f}'.format(
        #     discriminator_data_handler.acc_valid_loss[-1]))
        # print('--------------------------------------')

        return self.acc_valid_loss[-1], discriminator_data_handler.acc_valid_loss[-1]
