import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU

import torch_geometric.transforms as T
from torch_geometric.nn import GAE,GCNConv,SAGEConv, GINConv,GINEConv, GATConv, AGNNConv, GraphUNet
from torch_geometric.nn.models import InnerProductDecoder
from torch_geometric.utils import train_test_split_edges,dropout_adj

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

# GraphSage model def
class EncoderSAGE(torch.nn.Module):
    def __init__(self, in_channels, out_channels=256):
        super(EncoderSAGE, self).__init__()
        self.conv1 = SAGEConv(in_channels, 2 * out_channels)
        self.conv2 = SAGEConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        return self.conv2(x, edge_index)

# GIN model def
class EncoderGIN(torch.nn.Module):
    def __init__(self, in_channels, out_channels=32):
        super(EncoderGIN, self).__init__()
        
        nn1 = Sequential(Linear(in_channels, out_channels), ReLU(), Linear(out_channels, out_channels))
        self.conv1 = GINConv(nn1)
        self.bn1 = torch.nn.BatchNorm1d(out_channels)

        nn2 = Sequential(Linear(out_channels, out_channels), ReLU(), Linear(out_channels, out_channels))
        self.conv2 = GINConv(nn2)
        self.bn2 = torch.nn.BatchNorm1d(out_channels)

        nn3 = Sequential(Linear(out_channels, out_channels), ReLU(), Linear(out_channels, out_channels))
        self.conv3 = GINConv(nn3)
        self.bn3 = torch.nn.BatchNorm1d(out_channels)

        nn4 = Sequential(Linear(out_channels, out_channels), ReLU(), Linear(out_channels, out_channels))
        self.conv4 = GINConv(nn4)
        self.bn4 = torch.nn.BatchNorm1d(out_channels)

        nn5 = Sequential(Linear(out_channels, out_channels), ReLU(), Linear(out_channels, out_channels))
        self.conv5 = GINConv(nn5)
        self.bn5 = torch.nn.BatchNorm1d(out_channels)

    def forward(self, x, edge_index):
        x = F.elu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = F.elu(self.conv2(x, edge_index))
#         x = self.bn2(x)
        x = F.elu(self.conv3(x, edge_index))
        x = self.bn3(x)
        x = F.elu(self.conv4(x, edge_index))
#         x = self.bn4(x)
        x = F.elu(self.conv5(x, edge_index))
        x = self.bn5(x)
        x = F.normalize(x, eps=5e-4)
        return x

# GAT model def

class EncoderGAT(torch.nn.Module):
    def __init__(self, in_channels, out_channels=8,heads=8):
        super(EncoderGAT, self).__init__()
        self.conv1 = GATConv(in_channels, out_channels, heads=heads, dropout=0.6,concat=True)
        self.conv2 = GATConv(out_channels * heads, out_channels, heads=4,dropout=0.6,concat=True)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.normalize(self.conv2(x, edge_index),eps=5e-4)
        return x

# AGNN model def

class EncoderAGNN(torch.nn.Module):
    def __init__(self, in_channels, out_channels=16):
        super(EncoderAGNN, self).__init__()
        self.lin1 = torch.nn.Linear(in_channels, out_channels)
        self.prop1 = AGNNConv(requires_grad=True)
        self.prop2 = AGNNConv(requires_grad=True)

    def forward(self, x, edge_index):
#         x = F.dropout(x, training=self.training)
        x = F.relu(self.lin1(x))
        x = self.prop1(x, edge_index)
        x = self.prop2(x, edge_index)
        x = F.normalize(x,eps=5e-4)
        return x


# GraphUNet model def
class EncoderGraphUNet(torch.nn.Module):
    def __init__(self, in_channels,hidden_channels=16, out_channels=16):
        super(EncoderGraphUNet, self).__init__()
        
        self.unet = GraphUNet(in_channels=in_channels,
                              hidden_channels=hidden_channels,
                              out_channels=out_channels,
                              depth=4)

    def forward(self, x, edge_index):
        edge_index, _ = dropout_adj(edge_index, p=0.2,
                                    force_undirected=True,
                                    training=self.training)
        x = F.dropout(x, p=0.8, training=self.training)

        #x = self.unet(x, edge_index)
        x = F.normalize(self.unet(x, edge_index),eps=1e-3)
        return x  
    
class ComputeResult:
    def __init__(self, model_type='GCN', text_encoding='bert'):
        """
            Class for training N times and computing the results
            model_type: string. Model definition. Options {"GCN","SAGE", "GIN", "GAT", "AGNN","GraphUNet"} default GCN
            text_encoding: text representation. Options {"bert","tfidf","d2v"} default BERT
        """
        
        root =  'data/{}'.format(text_encoding)
        raw_path = '../../data/torch/{}/'.format(text_encoding) 
        dataset = ScisciDataset(root=root,raw_path = raw_path,transform=T.NormalizeFeatures())
        data = train_test_split_edges(dataset[0])
        self.model_type = model_type
        self.text_encoding = text_encoding
        self.root = root
        self.dataset = dataset
        self.data = data
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.patience = 100
        self.num_epochs = 450
    
    def time_mark(self):
        return datetime.datetime.now().strftime("%Y_%m_%d-%H:%M:%S")
        
    def fit_model_once(self):
        
        #GCN
        if self.model_type == "GCN":
            encoder = EncoderGCN(in_channels=self.dataset.num_features, out_channels=32)

        #SAGE
        if self.model_type == "SAGE":
            encoder = EncoderSAGE(in_channels=self.dataset.num_features, out_channels=32)

        #GIN
        if self.model_type == "GIN":
            encoder = EncoderGIN(in_channels=self.dataset.num_features, out_channels=32)

        #GAT
        if self.model_type == "GAT":
            encoder = EncoderGAT(in_channels=self.dataset.num_features, out_channels=16,heads=8)

        #AGNN
        if self.model_type == "AGNN":
            encoder = EncoderAGNN(in_channels=self.dataset.num_features, out_channels=16)

        #GraphUNet
        if self.model_type == "GraphUNet":
            encoder = EncoderGraphUNet(in_channels=self.dataset.num_features, hidden_channels=32, out_channels=16)
        
        model = GAE(encoder = encoder,decoder=InnerProductDecoder())

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        trainer = TrainerGae(model, self.device,self.data, writer_path='runs/{}/{}/'.format(self.model_type, self.text_encoding)+ self.time_mark())
        model = trainer.fit(optimizer,patience=self.patience,num_epochs=self.num_epochs)
        
        auc, ap = trainer.evaluate(validation=False, test=True)
        return model, auc, ap
    
    def write_results(self, auc, ap):
        path = 'results/{}_{}.csv'.format(self.model_type, self.text_encoding)

        #first time
        if not os.path.exists(path):
            with open(path, 'w') as file:
                file.write("auc,ap\n")

        with open(path, 'a') as file:
            file.write("{},{}\n".format(auc,ap))   
            
    def fit(self):
        for i in range(10):
            model, auc,ap = self.fit_model_once()
            torch.save(model.state_dict(), 'models/{}_{}_model.pt'.format(self.model_type, self.text_encoding))
            self.write_results(auc,ap)

    