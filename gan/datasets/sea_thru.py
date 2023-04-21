from parsers import SeaThruParser
from torch.utils.data import Dataset, DataLoader, random_split
import sys

sys.path.insert(1, "../")


class SeaThruDataset(Dataset):
    def __init__(self, params):
        super(Dataset, self).__init__()

        self.images = SeaThruParser(params["underwater"])
        self.length = len(self.images)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return self.images[index]


class SeaDataLoaderCreator:
    def __init__(self, params):
        self.params = params
        self.dataset = None

    def get_loaders(self):
        dataset = SeaThruDataset(self.params["datasets"])
        self.dataset = dataset

        training_len = int(dataset.length * self.params["train_percentage"])
        validation_len = len(dataset) - training_len

        if validation_len != 0:
            training_set, validation_set = random_split(
                dataset, [training_len, validation_len]
            )

            return DataLoader(
                dataset=training_set, **self.params["sea_data_loader"]
            ), DataLoader(dataset=validation_set, **self.params["sea_data_loader"])
        else:
            return DataLoader(dataset=dataset, **self.params["sea_data_loader"]), None


def get_sea_data(data, device):
    underwater = (data / 255).to(device)

    return underwater
