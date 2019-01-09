"""
Module defining classes that were used to conduct experiments in order to develop an encoder which encodes
linguistic information more incrementally and therefore closer to the way that humans process language.
"""

from machine.models.EncoderRNN import EncoderRNN
from machine.models.DecoderRNN import DecoderRNN
from machine.models.seq2seq import Seq2seq

import torch
from torch import nn
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EncoderRNNWrapper(EncoderRNN):
    def __init__(self, vocab_size, max_len, hidden_size, embedding_size, input_dropout_p=0, dropout_p=0, n_layers=1,
                 bidirectional=False, rnn_cell='gru', variable_lengths=False, **kwargs):
        super().__init__(vocab_size, max_len, hidden_size, embedding_size, input_dropout_p, dropout_p, n_layers,
                 bidirectional, rnn_cell, variable_lengths)


class IncrementalSeq2Seq(Seq2seq):
    """
    Extension of the Seq2Seq model class to enable more problem-specific capabilities.
    """
    def __init__(self, *args, use_embeddings=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_embeddings = use_embeddings

    def forward(self, input_variable, input_lengths=None, target_variable=None, teacher_forcing_ratio=0):
        encoder_outputs, encoder_hidden, encoder_predictions = self.encoder_module(input_variable, input_lengths)

        decoder_outputs, decoder_hidden, other = self.decoder_module(
            inputs=target_variable["decoder_output"],
            encoder_hidden=encoder_hidden,
            encoder_outputs=encoder_outputs,
            function=self.decode_function,
            teacher_forcing_ratio=teacher_forcing_ratio
        )

        # Add predictions of the encoder as well as the actual words in the input sequence to compute anticipation loss
        other["encoder_predictions"] = encoder_predictions
        other["shifted_input_variables"] = input_variable[:, 1:]

        return decoder_outputs, decoder_hidden, other


class AnticipatingEncoderRNN(EncoderRNN):
    """
    Special kind of encoder which tries to also predict the next token of the sequence, where wrong predictions are
    penalized with a loss function similar to the predictions of a decoder.
    """
    def __init__(self, vocab_size, max_len, hidden_size, embedding_size, input_dropout_p=0, dropout_p=0, n_layers=1,
                 bidirectional=False, rnn_cell='gru', variable_lengths=False, **kwargs):
        super().__init__(
            vocab_size, max_len, hidden_size, embedding_size, input_dropout_p, dropout_p, n_layers, bidirectional,
            rnn_cell, variable_lengths
        )
        self.prediction_layer = nn.Linear(hidden_size, vocab_size)  # Layer to predict next token in input sequence

    def forward(self, input_var, input_lengths=None):
        output, hidden = super().forward(input_var, input_lengths)

        # Try to predict next word in sequence
        encoder_predictions = []

        # output[:, :-1]: Don't use last output for prediction because there is no more token in the sequence that could
        # be predicted
        time_steps = [output] if output.shape[1] == 1 else output[:, :-1].split(1, dim=1)

        for o_t in time_steps:
            o_t = o_t.squeeze(1)
            predictive_dist = self.prediction_layer(o_t)
            predictive_dist = F.log_softmax(predictive_dist)
            encoder_predictions.append(predictive_dist)

        return output, hidden, encoder_predictions


class HierarchicalEncoderRNN:
    """
    Special kind of encoder which tries to implement the Chunk-and-Pass processing hypothesized by
    Christiansen & Chater (2016) for human language processing within the Now-or-Never bottleneck framework.

    First, a RNN is used to create the hidden states based on input tokens and previous hidden states.
    Then, a convolutional filter is used to create "chunks" which are then used to create another set of hidden
    representations.
    """
    def __init__(self, vocab_size, max_len, hidden_size, embedding_size, input_dropout_p=0, dropout_p=0, n_layers=1,
                 filter_size=2, bidirectional=False, rnn_cell='gru', variable_lengths=False, hierarchical_layers=2, **kwargs):

        assert hierarchical_layers > 1, "n_layers must be bigger than one, otherwise this model is just a normal encoder."

        self.encoding_layers = [
            EncoderRNN(vocab_size, max_len, hidden_size, embedding_size, input_dropout_p, dropout_p, 1,
                       bidirectional, rnn_cell, variable_lengths)
            for _ in range(hierarchical_layers)
        ]
        self.filter_size = filter_size
        self.variable_lengths = variable_lengths
        self.filters = [
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(filter_size, 1))
            for _ in range(hierarchical_layers - 1)  # Only necessary between layers
        ]

    def __call__(self, input_var, input_lengths=None):
        return self.forward(input_var, input_lengths)

    def forward(self, input_var, input_lengths=None):
        # Perform forward pass for bottom-most layer using input embeddings
        output, hidden = self.encoding_layers[0].forward(input_var, input_lengths)

        # For all other layers: Perform convolutions on hidden representations, then feed through next layer
        for encoding_layer, kernel in zip(self.encoding_layers[1:], self.filters):
            hidden_input = torch.cat(hidden, dim=0)
            num_hidden, batch_size, dim = hidden_input.size()
            hidden_input = hidden_input.view(batch_size, 1, num_hidden, dim)

            if num_hidden < self.filter_size:
                break  # If there aren't enough elements to perform convolution

            feature_maps = kernel(hidden_input)
            feature_maps = F.relu(feature_maps)
            feature_maps = feature_maps.squeeze(1)

            # Adjust size of convoluted sequences to avoid processing padding tokens
            input_lengths = [length - self.filter_size + 1 for length in input_lengths]

            if 0 in input_lengths:
                break

            output, hidden = self.encoding_layer_forward(encoding_layer, feature_maps, input_lengths)

        return output, hidden, None  # No predictions are being made here

    def encoding_layer_forward(self, encoding_layer, input_var, input_lengths=None):
        """
        Perform a complete forward pass with an encoder layer and arbitrary input (embeddings for the bottom-most layer
        or convoluted hidden states from the previous layer).
        """
        input_var = encoding_layer.input_dropout(input_var)
        if self.variable_lengths:
            input_var = nn.utils.rnn.pack_padded_sequence(
                input_var, input_lengths, batch_first=True)

        output, hidden = encoding_layer.rnn(input_var)

        if self.variable_lengths:
            output, _ = nn.utils.rnn.pad_packed_sequence(
                output, batch_first=True)

        return output, hidden


class BottleneckDecoderRNN(DecoderRNN):
    """
    Special kind of decoder which only has access to a certain part of the encoded sequence at every time step.
    """
    # TODO: Implement
    pass
