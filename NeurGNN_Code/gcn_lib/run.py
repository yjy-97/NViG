import math
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from torch_nn import BasicConv, batched_index_select, act_layer

def pairwise_distance(x):
    with torch.no_grad():
        x_inner = -2 * torch.matmul(x, x.transpose(2, 1))
        x_square = torch.sum(torch.mul(x, x), dim=-1, keepdim=True)
        return x_square + x_inner + x_square.transpose(2, 1)

def part_pairwise_distance(x, start_idx=0, end_idx=1):
    with torch.no_grad():
        x_part = x[:, start_idx:end_idx]
        x_square_part = torch.sum(torch.mul(x_part, x_part), dim=-1, keepdim=True)
        x_inner = -2 * torch.matmul(x_part, x.transpose(2, 1))
        x_square = torch.sum(torch.mul(x, x), dim=-1, keepdim=True)
        return x_square_part + x_inner + x_square.transpose(2, 1)

def xy_pairwise_distance(x, y):
    with torch.no_grad():
        xy_inner = -2 * torch.matmul(x, y.transpose(2, 1))
        x_square = torch.sum(torch.mul(x, x), dim=-1, keepdim=True)
        y_square = torch.sum(torch.mul(y, y), dim=-1, keepdim=True)
        return x_square + xy_inner + y_square.transpose(2, 1)

def dense_knn_matrix(x, k=16, relative_pos=None):
    with torch.no_grad():
        x = x.transpose(2, 1).squeeze(-1)
        print('x.shape', x.shape)
        batch_size, n_points, n_dims = x.shape

        n_part = 10000
        if n_points > n_part:
            nn_idx_list = []
            groups = math.ceil(n_points / n_part)
            for i in range(groups):
                start_idx = n_part * i
                end_idx = min(n_points, n_part * (i + 1))
                dist = part_pairwise_distance(x.detach(), start_idx, end_idx)
                _, nn_idx_part = torch.topk(-dist, k=k)
                nn_idx_list += [nn_idx_part]
            nn_idx = torch.cat(nn_idx_list, dim=1)
        else:
            dist = pairwise_distance(x.detach())
            if relative_pos is not None:
                dist += relative_pos
            _, nn_idx = torch.topk(-dist, k=k)

        center_idx = torch.arange(0, n_points, device=x.device).repeat(batch_size, k, 1).transpose(2, 1)
    return torch.stack((nn_idx, center_idx), dim=0)

def xy_dense_knn_matrix(x, y, k=16, relative_pos=None):
    with torch.no_grad():
        x = x.transpose(2, 1).squeeze(-1)
        y = y.transpose(2, 1).squeeze(-1)
        batch_size, n_points, n_dims = x.shape
        dist = xy_pairwise_distance(x.detach(), y.detach())
        if relative_pos is not None:
            dist += relative_pos
        _, nn_idx = torch.topk(-dist, k=k)
        center_idx = torch.arange(0, n_points, device=x.device).repeat(batch_size, k, 1).transpose(2, 1)
    return torch.stack((nn_idx, center_idx), dim=0)

class DenseDilated(nn.Module):
    def __init__(self, k=9, dilation=1, stochastic=False, epsilon=0.0):
        super(DenseDilated, self).__init__()
        self.dilation = dilation
        self.stochastic = stochastic
        self.epsilon = epsilon
        self.k = k

    def forward(self, edge_index):
        if self.stochastic:
            if torch.rand(1) < self.epsilon and self.training:
                num = self.k * self.dilation
                randnum = torch.randperm(num)[:self.k]
                edge_index = edge_index[:, :, :, randnum]
            else:
                edge_index = edge_index[:, :, :, ::self.dilation]
        else:
            edge_index = edge_index[:, :, :, ::self.dilation]
        return edge_index

class RegionGraphConstruction(nn.Module):
    def __init__(self, k=9, dilation=1, stochastic=False, epsilon=0.0):
        super(RegionGraphConstruction, self).__init__()
        self.dilation = dilation
        self.stochastic = stochastic
        self.epsilon = epsilon
        self.k = k
        self._dilated = DenseDilated(k, dilation, stochastic, epsilon)

    def forward(self, x, y=None, template=None, relative_pos=None):
        B, C, N, _ = x.shape
        template = template.reshape(1, 1, -1, 1).repeat(B, 1, 1, 1)
        assert template is not None, "template required!"
        mask = (template.squeeze(1).squeeze(-1) > 0)         

        out = torch.zeros(2, B, N, self.k, device=x.device, dtype=torch.long)

        x = F.normalize(x, p=2.0, dim=1)
        if y is not None:
            y = F.normalize(y, p=2.0, dim=1)

        for b in range(B):
            valid_idx = mask[b].nonzero(as_tuple=False).squeeze(-1)        
            if valid_idx.numel() == 0:
                continue

            x_sub = x[b:b+1, :, valid_idx, :]              

            if y is not None:
                y_sub = y[b:b+1]        
                print('x_sub',x_sub.shape)
                e = xy_dense_knn_matrix(x_sub, y_sub, self.k * self.dilation, relative_pos)
            else:
                e = dense_knn_matrix(x_sub, self.k * self.dilation, relative_pos)

            e = self._dilated(e)              

            out[0, b, valid_idx, :] = valid_idx[e[0,0]]                
            out[1, b, valid_idx, :] = valid_idx[e[1,0]]              

        return out

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
                                                           
        print(D)
        if D == 4:
            template = np.load(f"p28.npy")                 
            template = torch.tensor(template, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        elif D == 7:
            template = np.load(f"p14.npy")                 
            template = torch.tensor(template, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        elif D == 14:
            template = np.load(f"p7.npy")                 
            template = torch.tensor(template, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        else:
            template = np.load(f"p4.npy")                 
            template = torch.tensor(template, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        y = None
        if self.r > 1:
            y = F.avg_pool3d(x, self.r, self.r)
            y = y.reshape(B, C, -1, 1).contiguous()
        x = x.reshape(B, C, -1, 1).contiguous()
        print('test',x.shape)
        edge_index = self.dilated_knn_graph(x, y, template=template, relative_pos=relative_pos)
        x = super(BrainAwareDynamicGraphConv3d, self).forward(x, edge_index, y)
        return x.reshape(B, -1, D, H, W).contiguous()

def test_brain_aware_graph_conv():
    B = 4
    C = 128
    D = 28                                           
    H = 28
    W = 28

    x = torch.randn(B, C, D, H, W)                          

    relative_pos = None                                     

    conv_layer = BrainAwareDynamicGraphConv3d(in_channels=C, out_channels=C, kernel_size=9, dilation=1, r=1)

    output = conv_layer(x, relative_pos=relative_pos)

    print(f"Output shape: {output.shape}")

if __name__ == "__main__":
    test_brain_aware_graph_conv()
