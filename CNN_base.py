import torch
import torch.nn as nn
import torch.nn.functional as f


class CNNExtractor(nn.Module):
    def __init__(self, obs_space, feature_dim=32):
        super().__init__()
        C, H, W = obs_space.shape