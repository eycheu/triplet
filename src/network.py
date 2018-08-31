import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from torch.nn.init import xavier_uniform_
import torch.nn.functional as F

def clamp_by_norm(t, p=2, dim=1, max_norm=1.0):
    t_norm = t.norm(p=p, dim=dim, keepdim=True)
    clip_coef = max_norm / t_norm
    t_max_norm = torch.Tensor([max_norm])
    if clip_coef.is_cuda:
        t_max_norm = t_max_norm.cuda()
    clip_coef = torch.min(clip_coef, t_max_norm)
    return t.mul(clip_coef.expand_as(t))

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, padding, dropout=0.1):
        super(ResidualBlock, self).__init__()

        self.conv1 = weight_norm(nn.Conv2d(in_channels, 
                                           out_channels, 
                                           kernel_size=kernel_size,
                                           stride=stride, 
                                           padding=padding, 
                                           dilation=dilation))
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.selu1 = nn.SELU()
        self.dropout1 = nn.Dropout(dropout)

        
        self.conv2 = weight_norm(nn.Conv2d(out_channels, 
                                           out_channels, 
                                           kernel_size=kernel_size,
                                           stride=stride, 
                                           padding=padding, 
                                           dilation=dilation))
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.selu2 = nn.SELU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.bn1, self.selu1, self.dropout1,
                                 self.conv2, self.bn2, self.selu2, self.dropout2)
        self.downsample = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.selu = nn.SELU()
        self.init_weights()

    def init_weights(self):
        xavier_uniform_(self.conv1.weight.data)
        xavier_uniform_(self.conv2.weight.data)
        if self.downsample is not None:
            xavier_uniform_(self.downsample.weight.data)
    
    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.selu(out + res)

class ConvNet(nn.Module):
    def __init__(self, 
                 n_input_channels, 
                 list_level_channels, 
                 kernel_size=3, 
                 dropout=0.1):
        super(ConvNet, self).__init__()
        layers = []
        num_levels = len(list_level_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = n_input_channels if i == 0 else list_level_channels[i-1]
            out_channels = list_level_channels[i]
            layers += [ResidualBlock(in_channels, 
                                     out_channels, 
                                     kernel_size, 
                                     stride=1, 
                                     dilation=dilation_size,
                                     padding=int(((kernel_size-1) * dilation_size)/2), dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class NormEmbeddingNet(nn.Module):
    def __init__(self, 
                 list_channels, 
                 H, 
                 W,
                 kernel_size=3, 
                 dropout=0.1,
                 fc_nodes = [256, 256, 2],
                 p=2
                ):
        super(NormEmbeddingNet, self).__init__()
        
        self.convnet = ConvNet(list_channels[0],
                               list_channels[1:],
                               kernel_size=kernel_size,
                               dropout=dropout)
        
        self.linears = []
        for i in range(len(fc_nodes)):
            if i == 0:
                linear = nn.Linear(list_channels[-1] * H * W, fc_nodes[i])
            else:
                linear = nn.Linear(fc_nodes[i-1], fc_nodes[i])
            self.linears.append(linear)
            
        layers = []
        for i, linear in enumerate(self.linears):
            if i < (len(self.linears) - 1):
                layers.append(linear)
                layers.append(nn.BatchNorm1d(linear.out_features))
                layers.append(nn.SELU())
            else:
                layers.append(linear)
                
        self.fc = nn.Sequential(*layers)
        
        self.p = p
        
        self.init_weights()
    
    def init_weights(self):
        for linear in self.linears:
            xavier_uniform_(linear.weight.data)
        
    def forward(self, x):
        output = self.convnet(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        output = F.normalize(output, p=self.p, dim=1)
        return output

    def get_embedding(self, x):
        return self.forward(x)

    
class NormClampEmbeddingNet(nn.Module):
    def __init__(self, 
                 list_channels, 
                 H, 
                 W,
                 kernel_size=3, 
                 dropout=0.1,
                 fc_nodes = [256, 256, 2],
                 p=2
                ):
        super(NormClampEmbeddingNet, self).__init__()
        
        self.convnet = ConvNet(list_channels[0],
                               list_channels[1:],
                               kernel_size=kernel_size,
                               dropout=dropout)
        
        self.linears = []
        for i in range(len(fc_nodes)):
            if i == 0:
                linear = nn.Linear(list_channels[-1] * H * W, fc_nodes[i])
            else:
                linear = nn.Linear(fc_nodes[i-1], fc_nodes[i])
            self.linears.append(linear)
            
        layers = []
        for i, linear in enumerate(self.linears):
            if i < (len(self.linears) - 1):
                layers.append(linear)
                layers.append(nn.BatchNorm1d(linear.out_features))
                layers.append(nn.SELU())
            else:
                layers.append(linear)
                
        self.fc = nn.Sequential(*layers)
        
        self.p = p
        self.init_weights()
    
    def init_weights(self):
        for linear in self.linears:
            xavier_uniform_(linear.weight.data)
        
    def forward(self, x):
        output = self.convnet(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        # constrain embedding within unit norm
        output = clamp_by_norm(output, p=self.p, dim=1, max_norm=1.0)
        return output

    def get_embedding(self, x):
        return self.forward(x)


class SigmoidEmbeddingNet(nn.Module):
    def __init__(self, 
                 list_channels, 
                 H, 
                 W,
                 kernel_size=3, 
                 dropout=0.1,
                 fc_nodes = [256, 256, 2],
                ):
        super(SigmoidEmbeddingNet, self).__init__()
        
        self.convnet = ConvNet(list_channels[0],
                               list_channels[1:],
                               kernel_size=kernel_size,
                               dropout=dropout)
        
        self.linears = []
        for i in range(len(fc_nodes)):
            if i == 0:
                linear = nn.Linear(list_channels[-1] * H * W, fc_nodes[i])
            else:
                linear = nn.Linear(fc_nodes[i-1], fc_nodes[i])
            self.linears.append(linear)
            
        layers = []
        for i, linear in enumerate(self.linears):
            if i < (len(self.linears) - 1):
                layers.append(linear)
                layers.append(nn.SELU())
            else:
                layers.append(linear)
                layers.append(nn.Sigmoid())
                
        self.fc = nn.Sequential(*layers)
        self.init_weights()
    
    def init_weights(self):
        for linear in self.linears:
            xavier_uniform_(linear.weight.data)
        
    def forward(self, x):
        output = self.convnet(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output

    def get_embedding(self, x):
        return self.forward(x)
    
def extract_embedding(dataloader, model, use_cuda):
    with torch.no_grad():
        model.eval()
        embeddings = np.zeros((len(dataloader.dataset), 2))
        labels = np.zeros(len(dataloader.dataset))
        k = 0
        for data, target in dataloader:
            if use_cuda:
                data = data.cuda()
            embeddings[k:k+len(data)] = model.get_embedding(data).cpu().data.numpy()
            labels[k:k+len(data)] = target.numpy()
            k += len(data)
    return embeddings, labels

