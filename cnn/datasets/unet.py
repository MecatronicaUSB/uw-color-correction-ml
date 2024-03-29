from parsers import UWParser
from torch.utils.data import Dataset, DataLoader, random_split
import sys
import torchvision.transforms.functional as TF
import random

sys.path.insert(1, "../")


class UNETDataset(Dataset):
    def __init__(self, params):
        super(Dataset, self).__init__()

        dataset_path = params["datasets"]["synthetic"]

        self.images = UWParser(
            dataset_path + "/images",
            dataset_path + "/gt",
        )
        self.length = len(self.images)
        print("\nLoaded {0} images from {1}".format(self.length, dataset_path))

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        image, gt = self.images[index]

        if random.random() > 0.5:
            image = TF.vflip(image)
            gt = TF.vflip(gt)

        if random.random() > 0.5:
            image = TF.hflip(image)
            gt = TF.hflip(gt)

        return image / 255, gt / 255


class DataLoaderCreator:
    def __init__(self, params):
        self.params = params

    def get_loaders(self):
        dataset = UNETDataset(self.params)

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
