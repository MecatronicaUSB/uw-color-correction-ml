import numpy as np


class DataHandler():
    def __init__(self):
        # Values just for the epoch
        self.generator_loss = np.array([])
        self.discriminator_loss = np.array([])

        self.discriminator_accuracy_real = np.array([])
        self.discriminator_accuracy_fake = np.array([])

        # Accumulated values along the epochs
        self.generator_acc_loss = np.array([])
        self.discriminator_acc_loss = np.array([])

        self.discriminator_acc_accuracy_real = np.array([])
        self.discriminator_acc_accuracy_fake = np.array([])

    def epoch_end(self, epoch):
        # Calculate the mean loss for the epoch
        self.calculate_mean_data()

        # Reset the losses
        self.reset_data()

        # Get the mean loss for the epoch
        epoch_generator_loss = self.generator_acc_loss[-1]
        epoch_discriminator_loss = self.discriminator_acc_loss[-1]

        # Get the mean accuracy for the epoch
        epoch_acc_on_real = self.discriminator_acc_accuracy_real[-1]
        epoch_acc_on_fake = self.discriminator_acc_accuracy_fake[-1]

        # Print the mean loss for the epoch
        print('Generator cost: {0:.4f}'.format(epoch_generator_loss))
        print('Discriminator cost: {0:.4f}'.format(
            epoch_discriminator_loss))

        # ---------- Printing Discriminator accuracy
        print('\nDiscriminator Accuracy on Real images: {:.2f}%'.format(
            epoch_acc_on_real * 100))
        print('Discriminator Accuracy on Fake images: {:.2f}%'.format(
            epoch_acc_on_fake * 100))

        return epoch_generator_loss, epoch_discriminator_loss, epoch_acc_on_real, epoch_acc_on_fake

    def calculate_mean_data(self):
        # Calculate the mean loss for the epoch
        generator_loss = np.mean(self.generator_loss)
        discriminator_loss = np.mean(self.discriminator_loss)

        # Append the mean loss to the list of losses
        self.generator_acc_loss = np.append(
            self.generator_acc_loss, generator_loss)
        self.discriminator_acc_loss = np.append(
            self.discriminator_acc_loss, discriminator_loss)

        # Calculate the mean accuracy for the epoch
        discriminator_accuracy_real = np.mean(self.discriminator_accuracy_real)
        discriminator_accuracy_fake = np.mean(self.discriminator_accuracy_fake)

        # Append the mean accuracy to the list of accumulated
        self.discriminator_acc_accuracy_real = np.append(
            self.discriminator_acc_accuracy_real, discriminator_accuracy_real)
        self.discriminator_acc_accuracy_fake = np.append(
            self.discriminator_acc_accuracy_fake, discriminator_accuracy_fake)

    def append_loss(self, loss, type):
        if type == "generator":
            self.generator_loss = np.append(self.generator_loss, loss)
        elif type == "discriminator":
            self.discriminator_loss = np.append(self.discriminator_loss, loss)
        else:
            raise ValueError(
                "Type must be either 'generator' or 'discriminator'")

    def append_accuracy(self, accuracy_on_real, accuracy_on_fake):
        self.discriminator_accuracy_real = np.append(
            self.discriminator_accuracy_real, accuracy_on_real)
        self.discriminator_accuracy_fake = np.append(
            self.discriminator_accuracy_fake, accuracy_on_fake)

    def reset_data(self):
        self.reset_losses()

    def reset_losses(self):
        self.generator_loss = np.array([])
        self.discriminator_loss = np.array([])
