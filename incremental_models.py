"""
Module defining classes that were used to conduct experiments in order to develop an encoder which encodes
linguistic information more incrementally and therefore closer to the way that humans process language.
"""

# STD
import random

# EXT
from machine.models.EncoderRNN import EncoderRNN
from machine.models.DecoderRNN import DecoderRNN
from machine.models.seq2seq import Seq2seq
from machine.models.attention import Attention
import numpy as np
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

    Bottleneck type "window" specifies that the decoder only has access to a window of encoder hidden states when using
    attention. E.g when using window size 1, the attention at time step t=3 only has access to the encoder hidden states
    at time steps 2-4.

    Bottleneck type "past" allows access to all encoder hidden states 1 ... t at decoding time step t.
    """
    def __init__(self, *args, use_attention=None, bottleneck_type="window", window_size=1,
                 **kwargs):
        assert bottleneck_type in ("window", "past"), \
            f"Invalid mode for bottleneck decoder found: {mode}, window or past expected."
        assert use_attention == "pre-rnn", "This decoder can only be used with pre-RNN attention."

        self.bottleneck_type = bottleneck_type
        self.window_size = window_size
        super().__init__(*args, **kwargs, use_attention=use_attention)

        if use_attention:
            self.attention = BottleneckAttention(self.hidden_size, self.attention_method,
                                                 bottleneck_type=bottleneck_type, window_size=window_size)
        else:
            self.attention = None

    def forward(self, inputs=None, encoder_hidden=None, encoder_outputs=None,
                function=F.log_softmax, teacher_forcing_ratio=0):

        ret_dict = dict()
        if self.use_attention:
            ret_dict[DecoderRNN.KEY_ATTN_SCORE] = list()

        inputs, batch_size, max_length = self._validate_args(inputs, encoder_hidden, encoder_outputs,
                                                             function, teacher_forcing_ratio)

        decoder_hidden = self._init_state(encoder_hidden)

        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        decoder_outputs = []
        sequence_symbols = []
        lengths = np.array([max_length] * batch_size)

        def decode(step, step_output, step_attn):
            decoder_outputs.append(step_output)
            if self.use_attention:
                ret_dict[DecoderRNN.KEY_ATTN_SCORE].append(step_attn)
            symbols = decoder_outputs[-1].topk(1)[1]
            sequence_symbols.append(symbols)

            eos_batches = symbols.data.eq(self.eos_id)
            if eos_batches.dim() > 0:
                eos_batches = eos_batches.cpu().view(-1).numpy()
                update_idx = ((lengths > step) & eos_batches) != 0
                lengths[update_idx] = len(sequence_symbols)
            return symbols

        # When we use pre-rnn attention we must unroll the decoder. We need to calculate the attention based on
        # the previous hidden state, before we can calculate the next hidden state.
        # We also need to unroll when we don't use teacher forcing. We need perform the decoder steps
        # one-by-one since the output needs to be copied to the input of the
        # next step.
        if self.use_attention == 'pre-rnn' or not use_teacher_forcing:
            unrolling = True
        else:
            unrolling = False

        assert unrolling, "This decoder only works when using unrolling."

        symbols = None
        for di in range(max_length):
            # We always start with the SOS symbol as input. We need to add extra dimension of length 1 for the number of decoder steps (1 in this case)
            # When we use teacher forcing, we always use the target input.
            if di == 0 or use_teacher_forcing:
                decoder_input = inputs[:, di].unsqueeze(1)
            # If we don't use teacher forcing (and we are beyond the first
            # SOS step), we use the last output as new input
            else:
                decoder_input = symbols

            # Perform one forward step
            decoder_output, decoder_hidden, step_attn = self.forward_step(decoder_input, decoder_hidden, encoder_outputs,
                                                                          function=function, timestep=di)
            # Remove the unnecessary dimension.
            step_output = decoder_output.squeeze(1)
            # Get the actual symbol
            symbols = decode(di, step_output, step_attn)

        ret_dict[DecoderRNN.KEY_SEQUENCE] = sequence_symbols
        ret_dict[DecoderRNN.KEY_LENGTH] = lengths.tolist()

        return decoder_outputs, decoder_hidden, ret_dict


class BottleneckAttention(Attention):
    """
    Define the attention mechanism used inside the bottleneck decoder, that only has access to a time-dependent set
    of encoder hidden states.
    """
    def __init__(self, *args, bottleneck_type, window_size):
        self.bottleneck_type = bottleneck_type
        self.window_size = window_size

        super().__init__(*args)

    def forward(self, decoder_states, encoder_states,
                **attention_method_kwargs):
        timestep = attention_method_kwargs["timestep"]
        batch_size = decoder_states.size(0)
        decoder_states.size(2)
        input_size = encoder_states.size(1)

        # Compute attention vals
        attn = self.method(decoder_states, encoder_states)

        # Apply mask based on bottleneck type
        mask = self.create_mask(timestep, attn)
        attn = attn * mask

        attn = F.softmax(attn.view(-1, input_size),
                         dim=1).view(batch_size, -1, input_size)

        # (batch, out_len, in_len) * (batch, in_len, dim) -> (batch, out_len, dim)
        context = torch.bmm(attn, encoder_states)

        return context, attn

    def create_mask(self, timestep, attention):
        mask = torch.zeros(attention.shape)
        num_encoder_hidden = attention.shape[2]

        if self.bottleneck_type == "past":
            mask[:, :, :timestep+1] = 1

        elif self.bottleneck_type == "window":
            left_bound = max(0, timestep - self.window_size)
            right_bound = min(num_encoder_hidden, timestep + self.window_size)

            mask[:, :, left_bound:right_bound] = 1

        return mask
