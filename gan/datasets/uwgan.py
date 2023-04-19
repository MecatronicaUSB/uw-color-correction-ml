from parsers import SeaThruDataset, NYUDepthDataset
from torch.utils.data import Dataset, DataLoader, random_split
import sys

sys.path.insert(1, "../")

from utils.torch_utils import add_channel_first


class UWGANDataset(Dataset):
    def __init__(self, params):
        super(Dataset, self).__init__()

        self.in_air = NYUDepthDataset(params["in-air"])
        self.underwater = SeaThruDataset(params["underwater"])

        # Get the length of the smallest dataset
        self.length = len(self.in_air)
        # self.length = 8

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        images = dict()
        images["in_air"] = self.in_air[index]
        images["underwater"] = self.underwater[index]

        return images


class DataLoaderCreator:
    def __init__(self, params):
        self.params = params
        self.dataset = None

    def get_loaders(self):
        dataset = UWGANDataset(self.params["datasets"])
        self.dataset = dataset

        training_len = int(dataset.length * self.params["train_percentage"])
        validation_len = len(dataset) - training_len

        if validation_len != 0:
            training_set, validation_set = random_split(
                dataset, [training_len, validation_len]
            )

            return DataLoader(
                dataset=training_set, **self.params["data_loader"]
            ), DataLoader(dataset=validation_set, **self.params["data_loader"])
        else:
            return DataLoader(dataset=dataset, **self.params["data_loader"]), None


def get_data(data, device):
    rgb = data["in_air"][:, :3, :, :]
    depth = data["in_air"][:, 3, :, :]

    rgb = (rgb / 255).to(device)
    depth = (add_channel_first(depth) / 10).to(device)

    underwater = (data["underwater"] / 255).to(device)

    return rgb, depth, underwater
