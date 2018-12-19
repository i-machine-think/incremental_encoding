"""
Implementing metrics in order to measure the degree to which a model processes inputs incrementally.
"""

# EXT
from machine.metrics.metrics import Metric
import torch
import torch.nn as nn


class AverageIntegrationRatio(Metric):
    """
    This ratio measure how much of the current encoder hidden state is based on the current input token
    and on the last hidden state.

    For any hidden state h_t and input embedding x_t, the ratio is computed by

    min(||h_t - x_t|| / ||h_t - h_t-1||, ||h_t - x_t|| / ||h_t - h_t-1||)

    which is then averaged over T-1 time steps. To apply this measure, the dimensionality
    has to be the same for hidden states and word embeddings alike.
    """
    _NAME = "Integration ratio"
    _SHORTNAME = "intratio"
    _INPUT = "seqlist"

    def __init__(self):
        super().__init__(self._NAME, self._SHORTNAME, self._INPUT)
        self.batch_ratio = 0

    def get_val(self):
        return self.batch_ratio

    def reset(self):
        self.batch_ratio = 0

    def eval_batch(self, outputs, targets):
        targets = targets["input_variables"]  # Input embeddings
        outputs = outputs["encoder_outputs"]  # Hidden states
        batch_size = targets.size(0)
        timesteps = targets.size(1)
        embedding_dim = targets.size(2)
        hidden_dim = outputs.size(2)
        ratios = torch.zeros(batch_size, timesteps - 1)

        # Check if dimensionality corresponds for embeddings and hidden states
        assert embedding_dim == hidden_dim, "This metric only work if input embeddings and hidden states are of the" \
                                            "same dimensionality"

        # Calculate ratios
        for t in range(1, timesteps):
            h_t, h_previous, x_t = outputs[:, t], outputs[:, t - 1], targets[:, t]

            delta_x = nn.PairwiseDistance(h_t - x_t)
            delta_h = nn.PairwiseDistance(h_t - h_previous)

            ratios[:, t - 1] = torch.min(delta_x, delta_h, dim=1)

        self.batch_ratio = ratios.view(1, batch_size * timesteps).sum() / (batch_size * timesteps)




