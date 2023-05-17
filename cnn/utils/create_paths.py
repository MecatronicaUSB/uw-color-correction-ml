import os


def create_saving_paths(params, create_evaluation_paths=False):
    # ---------- Creating paths for saved images and weights
    output_real_image_path = params["output_image"]["real"]["saving_path"]
    output_synthetic_image_path = params["output_image"]["synthetic"]["saving_path"]
    unet_saving_path = params["unet"]["saving_path"]
    stats_saving_path = params["output_stats"]["saving_path"]

    if not os.path.exists(output_real_image_path):
        os.makedirs(output_real_image_path)

    if not os.path.exists(output_synthetic_image_path):
        os.makedirs(output_synthetic_image_path)

    if not os.path.exists(unet_saving_path):
        os.makedirs(unet_saving_path)

    if not os.path.exists(stats_saving_path):
        os.makedirs(stats_saving_path)

    if create_evaluation_paths:
        output_underwater_a = params["output_evaluation"]["real-underwater-a"]
        output_underwater_b = params["output_evaluation"]["real-underwater-b"]
        output_underwater_c = params["output_evaluation"]["real-underwater-c"]

        if not os.path.exists(output_underwater_a):
            os.makedirs(output_underwater_a)

        if not os.path.exists(output_underwater_b):
            os.makedirs(output_underwater_b)

        if not os.path.exists(output_underwater_c):
            os.makedirs(output_underwater_c)
