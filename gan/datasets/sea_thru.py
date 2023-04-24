from parsers import SeaThruParser
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import sys

sys.path.insert(1, "../")


class SeaThruDataset(Dataset):
    def __init__(self, params, device):
        super(Dataset, self).__init__()

        self.images = SeaThruParser(params["underwater"])
        self.length = len(self.images)
        self.device = device
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
        image = self.images[index]

        if self.augmentation:
            image = self.transform(self.images[index])

        return (image / 255).to(self.device)


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
