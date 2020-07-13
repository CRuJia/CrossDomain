from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from data.dataset import SimpleDataset, SetDataset, MultiSetDataset, EpisodicBatchSampler, MultiEpisodicBatchSampler
from abc import abstractmethod



class DataManager:
    @abstractmethod
    def get_data_loader(self, datafile, aug):
        pass

class SimpleDataManager(DataManager):
    #TODO
    def __init__(self, image_size, batch_size):
        super(SimpleDataManager, self).__init__()
        self.batch_size = batch_size

    def get_data_loader(self, aug):
        pass

class SetDataManager(DataManager):
    def __init__(self, image_size, data_file, n_way, n_support, n_query, n_episode=100):
        super(SetDataManager, self).__init__()
        self.image_size = image_size
        self.n_way = n_way
        self.data_file = data_file
        self.batch_size = n_support + n_query
        # self.n_support = n_support
        self.n_episode = n_episode

    def get_data_loader(self,aug):
        transform = transforms.Compose([
        transforms.Resize((84,84)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010))
    ])
        if isinstance(self.data_file, list):
            dataset = MultiSetDataset(data_files=self.data_file, batch_size=self.batch_size, transform=transform)
            sampler = MultiEpisodicBatchSampler(dataset.lens(), self.n_way, self.n_episode)
        else:
            dataset = SetDataset(datafile=self.data_file, batch_size=self.batch_size, transform=transform)
            sampler = EpisodicBatchSampler(len(dataset), self.n_way, self.n_episode)
        data_loader_params = dict(batch_sampler = sampler, num_workers=4)
        data_loader = DataLoader(dataset, **data_loader_params)
        return data_loader

