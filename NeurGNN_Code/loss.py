import torch
import torch.nn as nn
import torch.nn.functional as F

class BrainRegionGeometryConstraintLoss(nn.Module):
    def __init__(self, alpha=1.0, normalize=True):
\
\
\
           
        super().__init__()
        self.alpha = alpha
        self.normalize = normalize

    def forward(self, feat, labels):
\
\
\
           
        B, C = feat.shape[:2]

        z = feat.mean(dim=(2, 3, 4))          

        if self.normalize:
            z = F.normalize(z, dim=1)

        unique_labels = torch.unique(labels)
        centers = {}

        for lbl in unique_labels:
            mask = labels == lbl
            centers[int(lbl.item())] = z[mask].mean(dim=0)

        intra_loss = 0.0
        for i in range(B):
            center = centers[int(labels[i].item())]
            intra_loss += torch.norm(z[i] - center, p=2) ** 2
        intra_loss = intra_loss / B

        inter_loss = 0.0
        lbls = list(centers.keys())
        num_pairs = 0

        for i in range(len(lbls)):
            for j in range(i + 1, len(lbls)):
                c1 = centers[lbls[i]]
                c2 = centers[lbls[j]]
                inter_loss += torch.norm(c1 - c2, p=2) ** 2
                num_pairs += 1

        if num_pairs > 0:
            inter_loss = inter_loss / num_pairs

        loss = intra_loss - self.alpha * inter_loss
        loss = torch.abs(loss)
        return loss

def test_brgc_loss():
    torch.manual_seed(0)

    B, C, D, H, W = 4, 1024, 4, 4, 4
    feat = torch.randn(B, C, D, H, W, requires_grad=True)
    labels = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    print(labels.shape)

    loss_fn = BrainRegionGeometryConstraintLoss(alpha=1.0, normalize=True)
    loss = loss_fn(feat, labels)

    print("GPG loss:", loss.item())

    loss.backward()
    print("Backward OK, grad norm:", feat.grad.norm().item())

if __name__ == "__main__":
    test_brgc_loss()
