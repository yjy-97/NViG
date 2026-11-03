import torch
from torch import nn
from .torch_nn import BasicConv, batched_index_select, act_layer
from .graph_construction import RegionGraphConstruction
import torch.nn.functional as F
from timm.models.layers import DropPath
import numpy as np

class MultiStatisticGraphConv3d(nn.Module):
    def __init__(self, in_channels, out_channels,
                 act='relu', norm=None, bias=True, eps=1e-6):
        super(MultiStatisticGraphConv3d, self).__init__()

        self.eps = eps
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.nn = BasicConv(
            [in_channels * 4, out_channels],
            act=act,
            norm=norm,
            bias=bias
        )

    def forward(self, x, edge_index, y=None):
        x_i = batched_index_select(x, edge_index[1])                 

        if y is not None:
            x_j = batched_index_select(y, edge_index[0])
        else:
            x_j = batched_index_select(x, edge_index[0])

        delta = x_j - x_i                 

        max_delta, _ = torch.max(delta, dim=-1, keepdim=True)               
        mean_delta = torch.mean(delta, dim=-1, keepdim=True)               

        var_delta = torch.mean(
            (delta - mean_delta) ** 2, dim=-1, keepdim=True
        )
        std_delta = torch.sqrt(var_delta + self.eps)                        

        out = torch.cat(
            [
                x,                              
                max_delta,
                mean_delta,
                std_delta
            ],
            dim=1
        )  
        return self.nn(out)

class EdgeConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True):
        super(EdgeConv3d, self).__init__()
        self.nn = BasicConv([in_channels * 2, out_channels], act, norm, bias)

    def forward(self, x, edge_index, y=None):
        x_i = batched_index_select(x, edge_index[1])
        if y is not None:
            x_j = batched_index_select(y, edge_index[0])
        else:
            x_j = batched_index_select(x, edge_index[0])
        max_value, _ = torch.max(self.nn(torch.cat([x_i, x_j - x_i], dim=1)), -1, keepdim=True)
        return max_value

class GraphSAGE(nn.Module):
    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True):
        super(GraphSAGE, self).__init__()
        self.nn1 = BasicConv([in_channels, in_channels], act, norm, bias)
        self.nn2 = BasicConv([in_channels * 2, out_channels], act, norm, bias)

    def forward(self, x, edge_index, y=None):
        if y is not None:
            x_j = batched_index_select(y, edge_index[0])
        else:
            x_j = batched_index_select(x, edge_index[0])
        x_j, _ = torch.max(self.nn1(x_j), -1, keepdim=True)
        return self.nn2(torch.cat([x, x_j], dim=1))

class GINConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True):
        super(GINConv3d, self).__init__()
        self.nn = BasicConv([in_channels, out_channels], act, norm, bias)
        eps_init = 0.0
        self.eps = nn.Parameter(torch.Tensor([eps_init]))

    def forward(self, x, edge_index, y=None):
        if y is not None:
            x_j = batched_index_select(y, edge_index[0])
        else:
            x_j = batched_index_select(x, edge_index[0])
        x_j = torch.sum(x_j, -1, keepdim=True)
        return self.nn((1 + self.eps) * x + x_j)

class GraphConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, conv='edge', act='relu', norm=None, bias=True):
        super(GraphConv3d, self).__init__()
        if conv == 'edge':
            self.gconv = EdgeConv3d(in_channels, out_channels, act, norm, bias)
        elif conv == 'mr':
            self.gconv = MultiStatisticGraphConv3d(in_channels, out_channels, act, norm, bias)
        elif conv == 'sage':
            self.gconv = GraphSAGE(in_channels, out_channels, act, norm, bias)
        elif conv == 'gin':
            self.gconv = GINConv3d(in_channels, out_channels, act, norm, bias)
        else:
            raise NotImplementedError('conv:{} is not supported'.format(conv))

    def forward(self, x, edge_index, y=None):
        return self.gconv(x, edge_index, y)

class BrainAwareDynamicGraphConv3d(GraphConv3d):
    def __init__(self, in_channels, out_channels, kernel_size=9, dilation=1, conv='edge', act='relu',
                 norm=None, bias=True, stochastic=False, epsilon=0.0, r=1):
       
        super(BrainAwareDynamicGraphConv3d, self).__init__(in_channels=in_channels, out_channels=out_channels,
                                            conv=conv, act=act, norm=norm, bias=bias)
        self.k = kernel_size
        self.d = dilation
        self.r = r
        self.dilated_knn_graph = RegionGraphConstruction(kernel_size, dilation, stochastic, epsilon)

    def forward(self, x, relative_pos=None):
        B, C, D, H, W = x.shape
   
        if D == 4:
            template = np.load(r"/home/yejiayu/Work/Project/TMI_129/gcn_lib/p28.npy")                 
            template = torch.tensor(template, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        elif D == 7:
            template = np.load(r"/home/yejiayu/Work/Project/TMI_129/gcn_lib/p14.npy")                 
            template = torch.tensor(template, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        elif D == 14:
            template = np.load(r"/home/yejiayu/Work/Project/TMI_129/gcn_lib/p7.npy")                 
            template = torch.tensor(template, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        else:
            template = np.load(r"/home/yejiayu/Work/Project/TMI_129/gcn_lib/p4.npy")                 
            template = torch.tensor(template, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        y = None
        if self.r > 1:
            y = F.avg_pool3d(x, self.r, self.r)
            y = y.reshape(B, C, -1, 1).contiguous()
        x = x.reshape(B, C, -1, 1).contiguous()
        edge_index = self.dilated_knn_graph(x, y, template=template, relative_pos=relative_pos)
        x = super(BrainAwareDynamicGraphConv3d, self).forward(x, edge_index, y)
        return x.reshape(B, -1, D, H, W).contiguous()

class GraphPerceptionBlock(nn.Module):
    def __init__(self, in_channels, kernel_size=9, dilation=1, conv='edge', act='relu', norm=None,
                 bias=True, stochastic=False, epsilon=0.0, r=1, n=196, drop_path=0.0, relative_pos=False):
        super(GraphPerceptionBlock, self).__init__()
        self.channels = in_channels
        self.n = n
        self.r = r
        self.fc1 = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, 1, stride=1, padding=0),
            nn.BatchNorm3d(in_channels),
        )
        self.graph_conv = BrainAwareDynamicGraphConv3d(in_channels, in_channels * 2, kernel_size, dilation, conv,
                                        act, norm, bias, stochastic, epsilon, r)
        self.fc2 = nn.Sequential(
            nn.Conv3d(in_channels * 2, in_channels, 1, stride=1, padding=0),
            nn.BatchNorm3d(in_channels),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.relative_pos = None

    def _get_relative_pos(self, relative_pos, D, H, W):
        return relative_pos

    def forward(self, x):
        _tmp = x
        x = self.fc1(x)
        B, C, D, H, W = x.shape
        relative_pos = self._get_relative_pos(self.relative_pos, D, H, W)
        x = self.graph_conv(x, relative_pos)
        x = self.fc2(x)
        x = self.drop_path(x) + _tmp
        return x
