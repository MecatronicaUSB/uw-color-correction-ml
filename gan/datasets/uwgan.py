from torch.utils.data import Dataset, DataLoader, random_split
import sys
sys.path.insert(1, '../')
from parsers import SeaThruDataset, NYUDepthDataset

# TODO: Make a test
class UWGANDataset(Dataset):
    def __init__(self, params):
        super(Dataset, self).__init__()

        self.in_air = NYUDepthDataset(params['nyu-depth-v2'])
        self.underwater = SeaThruDataset(params['sea-thru-d1'])

        # Get the length of the largets dataset
        # self.length = max(len(self.in_air), len(self.underwater))
        self.length = 1000
        
    def __len__(self):
        return self.length

    def __getitem__(self, index):
        images = dict()
        images['in_air'] = self.in_air[index]
        images['underwater'] = self.underwater[index]

        return images

class DataLoaderCreator():
    def __init__(self, params):
        self.params = params

    def get_loaders(self):
        dataset = UWGANDataset(self.params['datasets'])

        training_len = int(dataset.length * 0.85)
        validation_len = len(dataset) - training_len

        training_set, validation_set = random_split(dataset, [training_len, validation_len])

        return DataLoader(dataset=training_set, **self.params['data_loader']), DataLoader(dataset=validation_set, **self.params['data_loader'])