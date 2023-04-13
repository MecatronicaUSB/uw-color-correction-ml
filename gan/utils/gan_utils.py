def handle_training_switch(generator, discriminator, acc_on_fake, params):
    # If we are training the generator, we do it until acc on fake images is less or equal than a certain %
    if generator.training and acc_on_fake <= params["gan_switch"]["lower_bound"]:
        # Return Generator now training False, Discriminator now training True
        print("Switching training mode to Discriminator")
        return False, True

    # If we are training the discriminator, we do it until acc on fake images is greater or equal than a certain %
    elif discriminator.training and acc_on_fake >= params["gan_switch"]["upper_bound"]:
        # Return Generator now training True, Discriminator now training False
        print("Switching training mode to Generator")
        return True, False

    return None, None
