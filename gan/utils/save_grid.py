import torchvision

def save_grid(images, path, nrow):
    grid_image = torchvision.utils.make_grid(images.cpu(), nrow=nrow)
    torchvision.utils.save_image(grid_image, path + ".jpg")