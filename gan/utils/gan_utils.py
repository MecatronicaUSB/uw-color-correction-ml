def handle_training_switch(is_generator_training, is_discriminator_training, acc_on_fake, params):
    # If we are training the generator, we do it until acc on fake images is less or equal than a certain %
    if is_generator_training and acc_on_fake <= params["gan_switch"]["lower_bound"]:
        # Return Generator now training False, Discriminator now training True and print text
        return False, True, "\n------ Training Discriminator"

    # If we are training the discriminator, we do it until acc on fake images is greater or equal than a certain %
    elif is_discriminator_training and acc_on_fake >= params["gan_switch"]["upper_bound"]:
        # Return Generator now training True, Discriminator now training False and print text
        return True, False, "\n------ Training Generator"

    else:
        return None, None, None
