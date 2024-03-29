from parsers import SeaThruParser
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import sys

sys.path.insert(1, "../")


class SeaThruDataset(Dataset):
    def __init__(self, params):
        super(Dataset, self).__init__()

        self.images = SeaThruParser(params["datasets"]["underwater"])
        self.length = len(self.images)
        self.length = 1449
        self.augmentation = params["sea_data_loader"]["augmentation"]

        self.transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
            ]
        )

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        image = self.images[index] / 255

        if self.augmentation:
            image = self.transform(self.images[index])

        return image


class SeaDataLoaderCreator:
    def __init__(self, params):
        self.params = params
        self.dataset = None

    def get_loaders(self):
        dataset = SeaThruDataset(self.params)
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
