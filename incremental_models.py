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
        if not self.use_embeddings:
            other["encoder_predictions"] = encoder_predictions
            other["shifted_input_variables"] = input_variable[:, 1:]

        # Use embedding and calculate a MSE loss instead
        else:
            other["encoder_predicted_embeddings"] = encoder_predictions
            other["shifted_input_embeddings"] = self.encoder.embedding(input_variable[:, 1:])

        return decoder_outputs, decoder_hidden, other


class AnticipatingEncoderRNN(EncoderRNN):
    """
    Special kind of encoder which tries to also predict the next token of the sequence, where wrong predictions are
    penalized with a loss function similar to the predictions of a decoder.
    """
    def __init__(self, vocab_size, max_len, hidden_size, embedding_size, input_dropout_p=0, dropout_p=0, n_layers=1,
                 bidirectional=False, rnn_cell='gru', variable_lengths=False):
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


class AnticipatingEmbeddingEncoderRNN(AnticipatingEncoderRNN):
    """
    Special kind of encoder which tries to also predict the next token of the sequence, but using the index of the
    predicted word to look up an embedding an impose the loss as the distance between the predicted embedding an the
    actual next embedding.
    """
    def forward(self, input_var, input_lengths=None):
        # Get output of AnticipatingEncoderRNN
        output, hidden, encoder_prediction_dists = super().forward(input_var, input_lengths)

        # Get indices of predicted words
        encoder_predicted_indices = [predictive_dist.argmax(dim=1) for predictive_dist in encoder_prediction_dists]

        # Look up embeddings
        encoder_predicted_embeddings = [self.embedding(indices) for indices in encoder_predicted_indices]

        return output, hidden, encoder_predicted_embeddings


class BottleneckDecoderRNN(DecoderRNN):
    """
    Special kind of decoder which only has access to a certain part of the encoded sequence at every time step.
    """
    # TODO: Implement
    pass
