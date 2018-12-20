"""
Implementing metrics in order to measure the degree to which a model processes inputs incrementally.
"""

# EXT
from machine.metrics.metrics import Metric
import torch
import torch.nn.functional as F


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
        hidden = targets["encoder_hidden"]  # Input embeddings
        embeddings = targets["encoder_embeddings"]  # Hidden states
        hidden = torch.cat(hidden, dim=0)
        timesteps, batch_size, hidden_dim = hidden.size()
        hidden = hidden.view(batch_size, timesteps, hidden_dim)  # Reshape into more intuitive order
        embedding_dim = embeddings.size(2)
        ratios = torch.zeros(batch_size, timesteps - 1)

        # Check if dimensionality corresponds for embeddings and hidden states
        assert embedding_dim == hidden_dim, "This metric only work if input embeddings and hidden states are of the" \
                                            "same dimensionality"

        # Calculate ratios
        for t in range(1, timesteps):
            h_t, h_previous, x_t = hidden[:, t], hidden[:, t - 1], embeddings[:, t]

            delta_x = F.pairwise_distance(h_t, x_t, p=2, keepdim=True)
            delta_h = F.pairwise_distance(h_t, h_previous, p=2, keepdim=True)

            compared = torch.cat((delta_x / delta_h, delta_h / delta_x), dim=1)
            minimum_score, _ = torch.min(compared, dim=1)
            ratios[:, t - 1] = minimum_score

        self.batch_ratio = ratios.view(batch_size * (timesteps - 1)).sum() / (batch_size * timesteps)




