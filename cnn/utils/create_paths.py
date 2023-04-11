import os


def create_saving_paths(params):
    # ---------- Creating paths for saved images and weights
    output_real_image_path = params["output_image"]["real"]["saving_path"]

    if not os.path.exists(output_real_image_path):
        os.makedirs(output_real_image_path)

    output_synthetic_image_path = params["output_image"]["synthetic"]["saving_path"]

    if not os.path.exists(output_synthetic_image_path):
        os.makedirs(output_synthetic_image_path)

    unet_saving_path = params["unet"]["saving_path"]

    if not os.path.exists(unet_saving_path):
        os.makedirs(unet_saving_path)
