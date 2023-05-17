from parsers import EvaluateParser
from torch.utils.data import Dataset, DataLoader
import sys

sys.path.insert(1, "../")


class EvaluateDataset(Dataset):
    def __init__(self, dataset_path):
        super(Dataset, self).__init__()

        self.images = EvaluateParser(dataset_path)
        self.length = len(self.images)
        print("\nLoaded {0} images from {1}".format(self.length, dataset_path))

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        image = self.images[index]

        return image / 255


class EvaluateDataLoaderCreator:
    def __init__(self, params, dataset_path):
        self.params = params
        self.dataset_path = dataset_path

    def get_loaders(self):
        dataset = EvaluateDataset(self.dataset_path)

        return DataLoader(dataset=dataset, **self.params["data_loader"]), None
