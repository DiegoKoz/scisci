import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm.notebook import tqdm
import numpy as np
#!pip install livelossplot --quiet
from livelossplot import PlotLosses
from .EarlyStopping import EarlyStopping

class TrainerGae:
    
    def __init__(self, model, device,data, writer_path='runs'):
        
        self.model = model.to(device) #I send the network to CUDA
        self.x = data.x.to(device) 
        self.train_pos_edge_index = data.train_pos_edge_index.to(device)
        self.val_neg_edge_index = data.val_neg_edge_index.to(device)
        self.val_pos_edge_index = data.val_pos_edge_index.to(device)
        self.test_neg_edge_index = data.test_neg_edge_index.to(device)
        self.test_pos_edge_index = data.test_pos_edge_index.to(device)
        self.device = device
        self.writer = SummaryWriter(writer_path)
        self.data = data
        #tensorboard --logdir={writer_path} --bind_all
        
    def add_graph(self):
        self.writer.add_graph(self.model, [self.data.x, self.data.edge_index])
        self.writer.close()
        
    def train(self,optimizer):
        self.model.train()
        optimizer.zero_grad()
        z = self.model.encode(self.x, self.train_pos_edge_index)
        loss = self.model.recon_loss(z, self.train_pos_edge_index)
        loss.backward()
        optimizer.step()

    @torch.no_grad()
    def evaluate(self, validation=True,test=False):
        self.model.eval()
        z = self.model.encode(self.x, self.train_pos_edge_index)
        if validation:
            auc, ap = self.model.test(z, self.val_pos_edge_index,self.val_neg_edge_index)
        if test:
            auc, ap = self.model.test(z, self.test_pos_edge_index,self.test_neg_edge_index)
        return auc, ap
    
    def fit(self,optimizer,patience, num_epochs=200):
        
        liveloss = PlotLosses()    
        # initialize the early_stopping object
        early_stopping = EarlyStopping(patience=patience, verbose=True,metric ='auc')
        
        for epoch in tqdm(range(num_epochs)):
            logs = {}
            self.train(optimizer)
            val_auc, val_ap = self.evaluate(validation=True,test=False)
            
            logs['val_auc'] = val_auc
            logs['val_ap'] = val_ap

            liveloss.update(logs)
            liveloss.send()

            self.writer.add_scalar('val_auc', val_auc, epoch)
            self.writer.add_scalar('val_ap', val_ap, epoch)
               
            ### Add Early stop implementation
            # early_stopping needs the validation loss to check if it has decresed, 
            # and if it has, it will make a checkpoint of the current model
            early_stopping(val_auc, self.model)

            if early_stopping.early_stop:
                print("Early stopping")
                break
        # load the last checkpoint with the best model
        self.model.load_state_dict(torch.load('checkpoint.pt'))
        return  self.model