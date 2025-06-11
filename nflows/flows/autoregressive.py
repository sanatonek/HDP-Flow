"""Implementations of autoregressive flows."""

from torch.nn import functional as F
import torch

from nflows.distributions.normal import StandardNormal
from nflows.flows.base import Flow
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
from nflows.transforms.base import CompositeTransform
from nflows.transforms.normalization import BatchNorm
from nflows.transforms.permutations import RandomPermutation, ReversePermutation

# from snake.activations import Snake


class MaskedAutoregressiveFlow(Flow):
    """An autoregressive flow that uses affine transforms with masking.

    Reference:
    > G. Papamakarios et al., Masked Autoregressive Flow for Density Estimation,
    > Advances in Neural Information Processing Systems, 2017.
    """

    def __init__(
        self,
        features,
        hidden_features,
        num_layers,
        periodic,
        num_blocks_per_layer,
        context_features=None,
        use_residual_blocks=False,
        use_random_masks=False,
        use_random_permutations=False,
        activation=F.relu,
        dropout_probability=0.0,
        batch_norm_within_layers=False,
        batch_norm_between_layers=False,
    ):

        if use_random_permutations:
            permutation_constructor = RandomPermutation
        else:
            permutation_constructor = ReversePermutation

        layers = []
        for _ in range(num_layers):
            layers.append(permutation_constructor(features))
            layers.append(
                MaskedAffineAutoregressiveTransform(
                    features=features,
                    hidden_features=hidden_features,
                    context_features=context_features,
                    num_blocks=num_blocks_per_layer,
                    use_residual_blocks=use_residual_blocks,
                    random_mask=use_random_masks,
                    activation=activation,
                    dropout_probability=dropout_probability,
                    use_batch_norm=batch_norm_within_layers,
                )
            )
            if batch_norm_between_layers:
                layers.append(BatchNorm(features))

        if periodic:
            embedding_net = torch.nn.Sequential(Sine(1),
                                                torch.nn.Linear(1, context_features),
                                                Sine(context_features))
        else:
            embedding_net = torch.nn.Sequential(torch.nn.Linear(1, context_features),
                                                torch.nn.ReLU())

        super().__init__(
            transform=CompositeTransform(layers),
            distribution=StandardNormal([features]),
            embedding_net=embedding_net
        )


class Sine(torch.nn.Module):
    def __init__(self, feat_size):
        super().__init__()
        self.a = torch.nn.Parameter(torch.Tensor([0.1]*feat_size), requires_grad=True)

    def forward(self, x):
        # return torch.sin(0.5**x)
        return torch.sin(self.a*x)**2/self.a


