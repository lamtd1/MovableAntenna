import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from dataclasses import dataclass

@dataclass
class PPOConfig:
    learning_rate: float = 0.0007
    gamma: float = 0.99
    gae_lambda: float = 1.0
    clip_coef: float = 0.4
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    total_timesteps: int = 300_000
    num_steps: int = 1000  # n_steps from main.py
    batch_size: int = 64 # Default SB3
    n_epochs: int = 10   # Default SB3
    
    # Computed at runtime usually, but we can default them
    num_minibatches: int = 4 # 1000 / 4 = 250 batch size? Or 1000 / 64 ~ 15 batches. 
                             # SB3 does full batch or minibatches? SB3 uses batch_size=64. 
                             # If n_steps=1000, we have 1000 samples. 
                             # 1000 // 64 = 15 updates per epoch.

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self, observation_shape, action_shape):
        super().__init__()
        # observation_shape should be the shape of the FLATTENED observation
        obs_dim = np.prod(observation_shape)
        action_dim = np.prod(action_shape)

        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, action_dim), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)