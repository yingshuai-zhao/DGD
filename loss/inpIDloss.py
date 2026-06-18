import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from einops import rearrange, repeat

class InpaintingDiffusionLoss(nn.Module):
    def __init__(self,lambda_ID=1e-2,lambda_eu=1e-2,eps=1e-6):
        super(InpaintingDiffusionLoss, self).__init__()
        self.lambda_ID = lambda_ID
        self.lambda_eu = lambda_eu
        self.eps=eps
    def batch_angular_distance_general(self,embeddings1, embeddings2):
        norm1 = torch.norm(embeddings1, p=2, dim=1, keepdim=True)+self.eps  # (bs, 1)
        norm2 = torch.norm(embeddings2, p=2, dim=1, keepdim=True)+self.eps  # (bs, 1)
        embeddings1_norm = embeddings1 / norm1
        embeddings2_norm = embeddings2 / norm2
        cos_sim = torch.sum(embeddings1_norm * embeddings2_norm, dim=1)  # (bs,)
        cos_sim = torch.clamp(cos_sim, -1.0, 0.95)
        angles = torch.acos(cos_sim)  # (bs,)
        return 1/(angles+self.eps)
    def forward(self,out_ID=None,src_ID=None,timesteps=None):
        if timesteps is not None:
            cosine_coefficients = self.lambda_ID * (1 + torch.cos(torch.pi * timesteps / 250)) / 2
            idloss=(cosine_coefficients * self.batch_angular_distance_general(out_ID, src_ID)).mean()
        else:
            idloss = self.lambda_ID * self.batch_angular_distance_general(out_ID, src_ID).mean()
        return idloss 





