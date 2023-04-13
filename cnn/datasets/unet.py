from parsers import UWDataset
from torch.utils.data import Dataset, DataLoader, random_split
import sys

sys.path.insert(1, "../")


class UNETDataset(Dataset):
    def __init__(self, params):
        super(Dataset, self).__init__()

        dataset_path = params["datasets"]["synthetic"]

        self.images = UWDataset(
            dataset_path + "/images",
            dataset_path + "/gt",
        )
        self.length = len(self.images)
        print("\nLoaded {0} images from {1}".format(self.length, dataset_path))

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return self.images[index]


class DataLoaderCreator:
    def __init__(self, params):
        self.params = params

    def get_loaders(self):
        dataset = UNETDataset(self.params)

        training_len = int(dataset.length * self.params["train_percentage"])
        validation_len = len(dataset) - training_len

        training_set, validation_set = random_split(
            dataset, [training_len, validation_len]
        )

        return DataLoader(
            dataset=training_set, **self.params["data_loader"]
        ), DataLoader(dataset=validation_set, **self.params["data_loader"])


def get_data(data, device):
    image, gt = data

    image = image.to(device) / 255
    gt = gt.to(device) / 255

    return image, gt
