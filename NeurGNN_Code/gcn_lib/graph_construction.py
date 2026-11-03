import math
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

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
        batch_size, n_points, n_dims = x.shape

        k = min(k, n_points)
        if k <= 0:
            nn_idx = torch.zeros(batch_size, n_points, k, device=x.device, dtype=torch.long)
            center_idx = torch.arange(0, n_points, device=x.device).repeat(batch_size, k, 1).transpose(2, 1)
            return torch.stack((nn_idx, center_idx), dim=0)

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
        batch_size, n_points_x, n_dims_x = x.shape
        batch_size, n_points_y, n_dims_y = y.shape

        k = min(k, n_points_y, n_points_x)
        if k <= 0:
            nn_idx = torch.zeros(batch_size, n_points_x, k, device=x.device, dtype=torch.long)
            center_idx = torch.arange(0, n_points_x, device=x.device).repeat(batch_size, k, 1).transpose(2, 1)
            return torch.stack((nn_idx, center_idx), dim=0)

        dist = xy_pairwise_distance(x.detach(), y.detach())
        if relative_pos is not None:
            dist += relative_pos
        _, nn_idx = torch.topk(-dist, k=k)
        center_idx = torch.arange(0, n_points_x, device=x.device).repeat(batch_size, k, 1).transpose(2, 1)
    return torch.stack((nn_idx, center_idx), dim=0)

class DenseDilated(nn.Module):
    def __init__(self, k=9, dilation=1, stochastic=False, epsilon=0.0):
        super(DenseDilated, self).__init__()
        self.dilation = dilation
        self.stochastic = stochastic
        self.epsilon = epsilon
        self.k = k  
    def forward(self, edge_index):
        if edge_index.shape[-1] == 0:  
            return torch.zeros(edge_index.shape[0], edge_index.shape[1],
                               edge_index.shape[2], self.k,
                               device=edge_index.device, dtype=edge_index.dtype)

        input_k = edge_index.shape[-1]

        if self.stochastic:
            if torch.rand(1) < self.epsilon and self.training:
                if input_k >= self.k:
                    randnum = torch.randperm(input_k)[:self.k]
                    edge_index = edge_index[:, :, :, randnum]
                else:
                    padding = torch.zeros(edge_index.shape[0], edge_index.shape[1],
                                          edge_index.shape[2], self.k - input_k,
                                          device=edge_index.device, dtype=edge_index.dtype)
                    edge_index = torch.cat([edge_index, padding], dim=-1)
            else:
                if self.dilation == 1:
                    edge_index = edge_index[:, :, :, :self.k] if input_k > self.k else edge_index
                else:
                    sampled_indices = torch.arange(0, min(input_k, self.k * self.dilation),
                                                   step=self.dilation, device=edge_index.device)[:self.k]
                    edge_index = edge_index[:, :, :, sampled_indices]
        else:
            if self.dilation == 1:
                edge_index = edge_index[:, :, :, :self.k] if input_k > self.k else edge_index
            else:
                sampled_indices = torch.arange(0, min(input_k, self.k * self.dilation),
                                               step=self.dilation, device=edge_index.device)[:self.k]
                edge_index = edge_index[:, :, :, sampled_indices]

        if edge_index.shape[-1] < self.k:
            padding = torch.zeros(edge_index.shape[0], edge_index.shape[1],
                                  edge_index.shape[2], self.k - edge_index.shape[-1],
                                  device=edge_index.device, dtype=edge_index.dtype)
            edge_index = torch.cat([edge_index, padding], dim=-1)
        elif edge_index.shape[-1] > self.k:
            edge_index = edge_index[:, :, :, :self.k]

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

            valid_idx = valid_idx[valid_idx < N]
            if valid_idx.numel() == 0:
                continue

            Nv = len(valid_idx)

            x_sub = x[b:b + 1, :, valid_idx, :]  

            actual_k = min(self.k * self.dilation, Nv)
            if actual_k <= 0:
                print(f'Warning: actual_k={actual_k}, skipping batch {b}')
                continue

            if y is not None:
                y_sub = y[b:b + 1]  
                e = xy_dense_knn_matrix(x_sub, y_sub, actual_k, relative_pos)
            else:
                e = dense_knn_matrix(x_sub, actual_k, relative_pos)

            if e.shape[-1] == 0: 
                print(f'Warning: empty edge_index for batch {b}')
                continue

            e = self._dilated(e)  

            actual_output_k = e.shape[-1]

            if actual_output_k < self.k:
                padding_needed = self.k - actual_output_k
                e_padded = torch.zeros(2, 1, Nv, self.k, device=e.device, dtype=e.dtype)
                e_padded[:, :, :, :actual_output_k] = e
                e = e_padded
            elif actual_output_k > self.k:
                e = e[:, :, :, :self.k]

            e_neighbors = e[0, 0] 
            e_centers = e[1, 0]  

            if y is not None:
                e_neighbors = torch.clamp(e_neighbors, 0, N - 1)
                out[0, b, valid_idx, :] = e_neighbors
            else:
                e_neighbors = torch.clamp(e_neighbors, 0, Nv - 1)
                mapped_neighbors = valid_idx[e_neighbors]
                out[0, b, valid_idx, :] = mapped_neighbors.cuda(6)

            out[1, b, valid_idx, :] = valid_idx.unsqueeze(1).repeat(1, self.k).cuda(6)
        return out
