from parsers import NYUDepthParser
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import sys

sys.path.insert(1, "../")

from utils.torch_utils import add_channel_first


class NYUDataset(Dataset):
    def __init__(self, params, device):
        super(Dataset, self).__init__()

        self.images = NYUDepthParser(params["datasets"]["in-air"])
        self.length = len(self.images)
        self.device = device
        self.augmentation = params["nyu_data_loader"]["augmentation"]
        self.force_crop = params["nyu_data_loader"]["force_crop"]

        self.transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
            ]
        )
        self.CROPPING_PIXELS = 16

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        image = self.images[index]

        if self.augmentation:
            image = self.transform(self.images[index])

        rgb, depth = self.split_rgb_depth(image)

        if self.force_crop:
            rgb, depth = self.crop(rgb, depth)

        return rgb, depth

    def split_rgb_depth(self, image):
        rgb = image[:, :3, :, :]
        depth = image[:, 3, :, :]

        rgb = (rgb / 255).to(self.device)
        depth = add_channel_first(depth / 10).to(self.device)

        return rgb, depth

    def get_item_cropped(self, index):
        rgb, depth = self.split_rgb_depth(self.images[index])

        return self.crop(rgb, depth)

    def crop(self, rgb, depth):
        # ----- Crop pixels at the border of both images
        cropped_rgb = rgb[
            :,
            self.CROPPING_PIXELS : -self.CROPPING_PIXELS,
            self.CROPPING_PIXELS : -self.CROPPING_PIXELS,
        ]
        cropped_depth = depth[
            :,
            self.CROPPING_PIXELS : -self.CROPPING_PIXELS,
            self.CROPPING_PIXELS : -self.CROPPING_PIXELS,
        ]

        return cropped_rgb, cropped_depth


class NYUDataLoaderCreator:
    def __init__(self, params, device):
        self.params = params
        self.device = device
        self.dataset = None

    def get_loaders(self):
        dataset = NYUDataset(self.params, self.device)
        self.dataset = dataset

        training_len = int(dataset.length * self.params["train_percentage"])
        validation_len = len(dataset) - training_len

        data_loader_params = {
            "batch_size": self.params["nyu_data_loader"].get("batch_size", 8),
            "shuffle": self.params["nyu_data_loader"].get("shuffle", False),
            "num_workers": self.params["nyu_data_loader"].get("num_workers", 8),
        }

        if validation_len != 0:
            training_set, validation_set = random_split(
                dataset, [training_len, validation_len]
            )

            return DataLoader(dataset=training_set, **data_loader_params), DataLoader(
                dataset=validation_set, **data_loader_params
            )
        else:
            return DataLoader(dataset=dataset, **data_loader_params), None
