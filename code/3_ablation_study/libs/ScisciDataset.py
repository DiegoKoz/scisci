import torch
import pickle

from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
import os.path as osp

class ScisciDataset(InMemoryDataset):

    r"""The citation network dataset
    Args:
        root (string): Root directory where the dataset should be saved.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
    """

    
    def __init__(self, root,raw_path = '../../data/torch/', transform=None,pre_transform=None):
        
        self.raw_path = raw_path
        
        super(ScisciDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])


    @property
    def raw_dir(self):
        return self.raw_path

    @property
    def processed_dir(self):
        return osp.join(self.root, 'Scisci', 'processed')

    @property
    def raw_file_names(self):
        names = ['x', 'edge_pairs']
        return ['{}.p'.format(name) for name in names]

    @property
    def processed_file_names(self):
         return 'data.pt'


    def download(self):
        pass

    def process(self):

        x,edge_pairs = [pickle.load(open('{}/{}'.format(self.raw_dir, name), 'rb'))for name in self.raw_file_names]

        data = Data(x=x, edge_index=edge_pairs)        
        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])
