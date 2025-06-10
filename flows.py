import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from nflows.flows.autoregressive import MaskedAutoregressiveFlow


class WrappedMAF(MaskedAutoregressiveFlow):
    def __init__(self,
                n_blocks,
                input_size,
                hidden_size,
                n_hidden,
                cond_label_size=None,
                activation="ReLU",
                periodic=False,
                input_order="sequential",
                batch_norm=True):
        
        features = input_size
        hidden_features = hidden_size
        num_layers = n_blocks
        num_blocks_per_layer = n_hidden
        use_residual_blocks = False#True
        use_random_masks = False
        use_random_permutations = True
        if periodic:
            activation = Snake()
            context_features = input_size
        else:
            activation = eval('F.' + activation.lower())
            context_features = cond_label_size
        dropout_probability = 0.0
        batch_norm_within_layers = batch_norm
        batch_norm_between_layers = False

        super().__init__(features=features, 
                        hidden_features=hidden_features,
                        context_features=context_features,
                        num_layers=num_layers,
                        periodic=periodic,
                        num_blocks_per_layer=num_blocks_per_layer,
                        use_residual_blocks=use_residual_blocks,
                        use_random_masks=use_random_masks,
                        use_random_permutations=use_random_permutations,
                        activation=activation,
                        dropout_probability=dropout_probability,
                        batch_norm_within_layers=batch_norm_within_layers,
                        batch_norm_between_layers=batch_norm_between_layers)


class Snake(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Parameter(torch.Tensor([0.1]), requires_grad=True)

    def forward(self, x):
        return torch.sin(self.a*x)**2/self.a
