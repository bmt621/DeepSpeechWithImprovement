import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CNNLayerNorm(nn.Module):
    """Layer normalization built for cnns input"""
    def __init__(self, n_feats):
        super(CNNLayerNorm, self).__init__()
        self.layer_norm = nn.LayerNorm(n_feats)

    def forward(self, x):
        # x (batch, channel, feature, time)
        x = x.transpose(2, 3).contiguous() # (batch, channel, time, feature)
        x = self.layer_norm(x)
        return x.transpose(2, 3).contiguous() # (batch, channel, feature, time)
    
    
class ResidualCNN(nn.Module):
    """Residual CNN with layer norm instead of batch norm
    """
    def __init__(self, in_channels, out_channels, kernel, stride,n_feats, dropout:float=0.1):
        super(ResidualCNN, self).__init__()

        self.cnn1 = nn.Conv2d(in_channels, out_channels, kernel, stride, padding=kernel//2)
        self.cnn2 = nn.Conv2d(out_channels, out_channels, kernel, stride, padding=kernel//2)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        self.layer_norm1 = CNNLayerNorm(n_feats)
        self.layer_norm2 = CNNLayerNorm(n_feats)

    def forward(self, x):
        residual = x  # (batch, channel, feature, time)
        x = self.layer_norm1(x)
        x = F.gelu(x)
        x = self.dropout1(x)
        x = self.cnn1(x)
        x = self.layer_norm2(x)
        x = F.gelu(x)
        x = self.dropout2(x)
        x = self.cnn2(x)
        x += residual
        return x # (batch, channel, feature, time)
    
    
    
class RNN(nn.Module):
    def __init__(self,rnn_dim,hidden_size,batch_first,dropout=0.1):
        super(RNN,self).__init__()
        
        self.rnn = nn.GRU(input_size=rnn_dim,hidden_size=hidden_size,num_layers=2,\
                          dropout=dropout,batch_first=batch_first,bidirectional=True)
        
        self.layer_norm = nn.LayerNorm(rnn_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self,x):
        x = self.layer_norm(x)
        x = F.gelu(x)
        x,_ = self.rnn(x)
        x = self.dropout(x)
        
        return(x)
    
    
class SpeechModel(nn.Module):
    def __init__(self,n_rnn_layers,n_resnet_layer,n_rnn_dim,n_feats,n_class,stride=2,dropout=0.1):
        super(SpeechModel,self).__init__()
        
        self.n_feats = n_feats//2
        
        self.cnn = nn.Conv2d(1, 32, 3, stride=stride, padding=3//2)
        
        self.resnet=nn.Sequential(*[ResidualCNN(32,32, kernel=3,stride=1,dropout=dropout,n_feats=self.n_feats)
                                   for _ in range(n_resnet_layer)]
                                 )
        self.rnn = nn.Sequential(*[RNN(rnn_dim = n_rnn_dim if i ==0 else n_rnn_dim*2,
                                       hidden_size = n_rnn_dim, batch_first =i ==0 )
                                   for i in range(n_rnn_layers)])
        self.n_rnn_dim = n_rnn_dim
        
        
        self.clf = nn.Sequential(
            nn.Linear(n_rnn_dim*2, self.n_rnn_dim),  # birnn returns rnn_dim*2
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(n_rnn_dim, n_class)
        )
        
        
        self.fcc = nn.Linear(self.n_feats*32, self.n_rnn_dim)
        
        
    def forward(self,x):
        x = self.cnn(x)
        x = self.resnet(x)
        shape = x.shape
        x = x.view(shape[0],shape[1]*shape[2],shape[3]).transpose(1,2) # (batch, time, features)
        x = self.fcc(x)
        
        x = self.rnn(x)
        x = self.clf(x)
        
        return(x)