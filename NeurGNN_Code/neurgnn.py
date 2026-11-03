import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential as Seq

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import load_pretrained
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from gcn_lib.graph_processing import GraphPerceptionBlock, act_layer
import numpy as np

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 2, 'input_size': (1, 112, 112, 112), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }

default_cfgs = {
    'neurgnn_b_112_gelu': _cfg(
        crop_pct=0.95, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    )
}

class GraphFeedForwardBlock(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act='relu', drop_path=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Sequential(
            nn.Conv3d(in_features, hidden_features, 1, stride=1, padding=0),
            nn.Conv3d(hidden_features, hidden_features, 1, stride=1, padding=0, groups=4),
            nn.BatchNorm3d(hidden_features),
        )
        self.act = act_layer(act)
        self.fc2 = nn.Sequential(
            nn.Conv3d(hidden_features, out_features, 1, stride=1, padding=0),
            nn.Conv3d(out_features, out_features, 1, stride=1, padding=0, groups=4),
            nn.BatchNorm3d(out_features),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)

        x = self.drop_path(x) + shortcut
        return x

class NeuroimagingPatchEmbedding(nn.Module):
    def __init__(self, img_size=224, in_dim=1, out_dim=768, act='relu'):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv3d(in_dim, out_dim // 2, 3, stride=2, padding=1),
            nn.BatchNorm3d(out_dim // 2),
            act_layer(act),
            nn.Conv3d(out_dim // 2, out_dim, 3, stride=2, padding=1),
            nn.BatchNorm3d(out_dim),
            act_layer(act),
            nn.Conv3d(out_dim, out_dim, 3, stride=1, padding=1),
            nn.BatchNorm3d(out_dim),
        )

    def forward(self, x):
        x = self.convs(x)
        return x

class StageDownsampling3D(nn.Module):
    def __init__(self, in_dim=1, out_dim=768):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_dim, out_dim, 3, stride=2, padding=1),
            nn.BatchNorm3d(out_dim),
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class NeurGNN(torch.nn.Module):
    def __init__(self, opt):
        super(NeurGNN, self).__init__()
        k = opt.k
        act = opt.act
        norm = opt.norm
        bias = opt.bias
        epsilon = opt.epsilon
        stochastic = opt.use_stochastic
        conv = opt.conv
        emb_dims = opt.emb_dims
        drop_path = opt.drop_path
        in_dim = getattr(opt, "in_dim", 1)

        blocks = opt.blocks
        self.n_blocks = sum(blocks)
        channels = opt.channels
        reduce_ratios = [4, 2, 1, 1]
        dpr = [x.item() for x in torch.linspace(0, drop_path, self.n_blocks)]                               
        num_knn = [int(x.item()) for x in torch.linspace(k, k, self.n_blocks)]                     
        max_dilation = 49 // max(num_knn)

        self.stem = NeuroimagingPatchEmbedding(in_dim=in_dim, out_dim=channels[0], act=act)
                                                                                                  
        self.pos_embed = self.load_bspe_encoding(channels[0])
        HWD = 112 // 4 * 112 // 4 * 112 // 4

        self.backbone = nn.ModuleList([])
        idx = 0
        for i in range(len(blocks)):
            if i > 0:
                self.backbone.append(StageDownsampling3D(channels[i - 1], channels[i]))
            for j in range(blocks[i]):
                self.backbone += [
                    Seq(GraphPerceptionBlock(channels[i], num_knn[idx], min(idx // 4 + 1, max_dilation), conv, act, norm,
                            bias, stochastic, epsilon, reduce_ratios[i], n=HWD, drop_path=dpr[idx],
                            relative_pos=False),
                        GraphFeedForwardBlock(channels[i], channels[i] * 4, act=act, drop_path=dpr[idx])
                        )]
                idx += 1
        self.backbone = Seq(*self.backbone)

        self.prediction = Seq(nn.Conv3d(channels[-1], 1024, 1, bias=True),
                              nn.BatchNorm3d(1024),
                              act_layer(act),
                              nn.Dropout(opt.dropout),
                              nn.Conv3d(1024, opt.n_classes, 1, bias=True))
        self.initialize_weights()

    def load_bspe_encoding(self, chan):
        npy_file_path = r"/home/yejiayu/Work/Project/TMI_130/pos.npy"            
        data = np.load(npy_file_path)                                

        data = torch.tensor(data, dtype=torch.float32)             
        data = data.unsqueeze(0)                              
        expanded_data = data.repeat(1, chan, 1, 1, 1)
        return expanded_data

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True

    def forward(self, inputs):
        x = self.stem(inputs) + self.pos_embed.cuda(6)
        for i in range(len(self.backbone)):
            x = self.backbone[i](x)

        x = F.adaptive_avg_pool3d(x, 1)
        return self.prediction(x).squeeze(-1).squeeze(-1).squeeze(-1),x

@register_model
def neurgnn_b_112_gelu(pretrained=False, **kwargs):
    class OptInit:
        def __init__(self, num_classes=2, drop_path_rate=0.0, in_dim=1, **kwargs):
            self.k = 9
            self.conv = 'mr'
            self.act = 'gelu'
            self.norm = 'batch'
            self.bias = True
            self.dropout = 0.0
            self.use_dilation = True
            self.epsilon = 0.2
            self.use_stochastic = False
            self.drop_path = drop_path_rate
            self.blocks = [2, 2, 18, 2]
            self.channels = [128, 256, 512, 1024]
            self.n_classes = num_classes
            self.emb_dims = 1024
            self.in_dim = in_dim

    opt = OptInit(**kwargs)
    model = NeurGNN(opt)
    model.default_cfg = default_cfgs['neurgnn_b_112_gelu']
    return model

if __name__ == "__main__":
    model = neurgnn_b_112_gelu(in_dim=2)
    x = torch.rand(4, 2, 112, 112, 112)
    out = model(x)
    print(out.shape)
