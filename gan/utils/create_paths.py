import os


def create_saving_paths(params):
    # ---------- Creating paths for saved images and weights
    output_image_path = params["output_image"]["saving_path"]

    if not os.path.exists(output_image_path):
        os.makedirs(output_image_path)

    generator_saving_path = params["generator"]["saving_path"]

    if not os.path.exists(generator_saving_path):
        os.makedirs(generator_saving_path)

    discriminator_saving_path = params["discriminator"]["saving_path"]

    if not os.path.exists(discriminator_saving_path):
        os.makedirs(discriminator_saving_path)

    stats_saving_path = params["output_stats"]["saving_path"]

    if not os.path.exists(stats_saving_path):
        os.makedirs(stats_saving_path)
