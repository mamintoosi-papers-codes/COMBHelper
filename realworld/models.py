import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv
import torch.nn.functional as F
import numpy as np
from geomloss import SamplesLoss
# import torch_directml

# def select_device(device=''):
#     if device.lower() == 'cuda':
#         if not torch.cuda.is_available():
#             print ("torch.cuda not available")
#             return torch.device('cpu')    
#         else:
#             return torch.device('cuda:0')
#     if device.lower() == 'dml':
#         return torch_directml.device(torch_directml.default_device())
#     else:
#         return torch.device('cpu')
    
# 4 layers
class GCN1(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN1, self).__init__()
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels, aggr='sum', normalize=True))
        self.convs.append(SAGEConv(hidden_channels, hidden_channels, aggr='sum', normalize=True))
        self.convs.append(SAGEConv(hidden_channels, hidden_channels, aggr='sum', normalize=True))
        self.convs.append(SAGEConv(hidden_channels, out_channels, aggr='sum', normalize=True))
    
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
    
    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)
        return F.log_softmax(x, dim=-1)
    

# 3 layers
class GCN2(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN2, self).__init__()
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels, aggr='sum', normalize=True))
        self.convs.append(SAGEConv(hidden_channels, hidden_channels, aggr='sum', normalize=True))
        self.convs.append(SAGEConv(hidden_channels, out_channels, aggr='sum', normalize=True))
    
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
    
    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)
        return F.log_softmax(x, dim=-1)
    
    
class TeacherModel(nn.Module):
    def __init__(self, encoder):
        super(TeacherModel, self).__init__()
        self.encoder = encoder
    
    def reset_parameters(self):
        self.encoder.reset_parameters()
    
    def forward(self, data):
        out = self.encoder(data)
        out = out[data.train_mask]
        out = F.log_softmax(out, dim=-1)
        y = data.y[data.train_mask]
        loss_train = F.nll_loss(out, y)
        acc_train = int((out.argmax(dim=-1) == y).sum()) / len(y)
        return loss_train, acc_train
    
    @torch.no_grad()
    def validate(self, data, weights):
        out = self.encoder(data)
        out = F.log_softmax(out, dim=-1)
        y = data.y
        preds = out[data.val_mask].argmax(dim=-1)
        acc_val = int((preds == y[data.val_mask]).sum()) / len(y[data.val_mask])
        
        acc_train = int((out[data.train_mask].argmax(dim=-1) == y[data.train_mask]).sum()) / len(y[data.train_mask])
        
        error = 1 - acc_train
        alpha = 0.5 * np.log((1 - error) / error)
        updated_weights = weights.clone()
        updated_weights[out[data.train_mask].argmax(dim=-1) == y[data.train_mask]] *= np.exp(0 - alpha)
        updated_weights[out[data.train_mask].argmax(dim=-1) != y[data.train_mask]] *= np.exp(alpha)
        
        updated_weights = updated_weights / updated_weights.sum()
        updated_weights *= torch.exp(data.weight)
        
        return acc_val, updated_weights / updated_weights.sum()
    
    @torch.no_grad()
    def test(self, data):
        out = self.encoder(data)
        out = F.log_softmax(out, dim=-1)
        y = data.y
        acc_test = int((out.argmax(dim=-1) == y).sum()) / len(y)
        return acc_test
    

class StudentModel(nn.Module):
    def __init__(self, encoder_t, encoder_s, T, alpha, beta, boosting, num_class):
        super(StudentModel, self).__init__()
        self.encoder_t = encoder_t
        self.encoder_s = encoder_s
        self.T = T
        self.alpha = alpha
        self.beta = beta
        self.boosting = True if boosting == 1 else False
        self.num_class = num_class
            
        for para in self.encoder_t.parameters():
            para.requires_grad = False
            
    def reset_parameters(self):
        self.encoder_s.reset_parameters()
    
    def KD_loss(self, out_s, out_t):
        out_s = F.log_softmax(out_s / self.T, dim=-1)
        out_t = F.softmax(out_t / self.T, dim=-1)
        return F.kl_div(out_s, out_t, reduction='sum') * (self.T ** 2) / out_s.shape[0]
    
    
    def wasserstein_loss(self, out_s, out_t):
        # Define a Sinkhorn (~Wasserstein) loss between sampled measures
        out_s = F.log_softmax(out_s / self.T, dim=-1)
        out_t = F.softmax(out_t / self.T, dim=-1)
        loss = SamplesLoss(loss="sinkhorn", p=2, blur=.05)
        return loss(out_s, out_t)

        # out_s = F.log_softmax(out_s / self.T, dim=-1)
        # out_t = F.softmax(out_t / self.T, dim=-1)
        # return torch.mean(out_s * out_t) / (self.T ** 2)

    def forward(self, data, weights):
        out_s = self.encoder_s(data)
        with torch.no_grad():
            out_t = self.encoder_t(data).detach()
        
        out_s = out_s[data.train_mask]
        out_t = out_t[data.train_mask]
        
        kd_loss = self.KD_loss(out_s, out_t)
        
        y = data.y[data.train_mask]
        
        out_s = F.log_softmax(out_s, dim=-1)
        acc_train = int((out_s.argmax(dim=-1) == y).sum()) / len(y)
        
        if self.boosting:
            clf_loss = (F.nll_loss(out_s, y, reduction='none') * weights).sum()
        else:
            clf_loss = F.nll_loss(out_s, y)
        # print(out_s.shape, y.shape)
        # wass_loss = self.wasserstein_loss(out_s, out_t)
        # SamplesLoss("sinkhorn", p=2, blur=0.01)
        loss_train = self.alpha * clf_loss + self.beta * kd_loss #+ 0.001*wass_loss
        return loss_train, acc_train
    
    @torch.no_grad()
    def validate(self, data, weights):
        out = self.encoder_s(data)
        out = F.log_softmax(out, dim=-1)
        y = data.y
        preds = out[data.val_mask].argmax(dim=-1)
        acc_val = int((preds == y[data.val_mask]).sum()) / len(y[data.val_mask])
        
        acc_train = int((out[data.train_mask].argmax(dim=-1) == y[data.train_mask]).sum()) / len(y[data.train_mask])
        
        error = 1 - acc_train
        alpha = 0.5 * np.log((1 - error) / error)
        updated_weights = weights.clone()
        updated_weights[out[data.train_mask].argmax(dim=-1) == y[data.train_mask]] *= np.exp(0 - alpha)
        updated_weights[out[data.train_mask].argmax(dim=-1) != y[data.train_mask]] *= np.exp(alpha)
        
        return acc_val, updated_weights / updated_weights.sum()
    
    @torch.no_grad()
    def test(self, data):
        out = self.encoder_s(data)
        out = F.log_softmax(out, dim=-1)
        y = data.y
        acc_test = int((out.argmax(dim=-1) == y).sum()) / len(y)
        return acc_test