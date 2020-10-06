import torch
import torch.nn.functional as F
# from torch.nn import Sequential, Linear, ReLU

import torch_geometric.transforms as T
from torch_geometric.nn import GAE,GCNConv
from torch_geometric.nn.models import InnerProductDecoder
from torch_geometric.utils import train_test_split_edges

import datetime
import os

from libs.TrainerGae import TrainerGae
from libs.ScisciDataset import ScisciDataset

from tqdm import tqdm

torch.manual_seed(12345)

## Models definitions

# GCN model def
class EncoderGCN(torch.nn.Module):
    def __init__(self, in_channels, out_channels=16):
        super(EncoderGCN, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels, cached=False)
        self.conv2 = GCNConv(2 * out_channels, out_channels, cached=False)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        return self.conv2(x, edge_index)

class ComputeResult:
    def __init__(self, feature):
        root = raw_path =  'data/{}'.format(feature)
        dataset = ScisciDataset(root=root,raw_path = raw_path,transform=T.NormalizeFeatures())
        data = train_test_split_edges(dataset[0])
        self.feature = feature
        self.root = root
        self.dataset = dataset
        self.data = data
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.patience = 100
        self.num_epochs = 400
    
    def time_mark(self):
        return datetime.datetime.now().strftime("%Y_%m_%d-%H:%M:%S")
        
    def fit_model_once(self):
        gcn_model = GAE(encoder = EncoderGCN(in_channels=self.dataset.num_features, out_channels=32),
                        decoder=InnerProductDecoder())

        optimizer = torch.optim.Adam(gcn_model.parameters(), lr=0.01)

        trainer_gcn = TrainerGae(gcn_model, self.device,self.data, writer_path='runs/gae_gcn/{}/'.format(self.feature)+ self.time_mark())
        gcn_model = trainer_gcn.fit(optimizer,patience=self.patience,num_epochs=self.num_epochs)

        auc, ap = trainer_gcn.evaluate(validation=False, test=True)
        return auc, ap
    
    def write_results(self, auc, ap):
        path = 'results/{}.csv'.format(self.feature)

        #first time
        if not os.path.exists(path):
            with open(path, 'w') as file:
                file.write("auc,ap\n")

        with open(path, 'a') as file:
            file.write("{},{}\n".format(auc,ap))   
            
    def fit(self):
        for i in range(10):
            auc,ap = self.fit_model_once()
            self.write_results(auc,ap)

    