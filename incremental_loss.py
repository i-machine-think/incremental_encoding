from __future__ import print_function
import math
import torch.nn as nn
import numpy as np
from torch import autograd
import torch


from machine.loss.loss import NLLLoss, Loss


class AnticipationLoss(NLLLoss):
    """
    Loss imposed on an extended encoder that also tries to predict the next word in the sequence.
    """
    _NAME = "Anticipation Loss"
    _SHORTNAME = "antcp_loss"
    _INPUTS = "encoder_predictions"
    _TARGETS = "shifted_input_variables"

    def eval_batch(self, decoder_outputs, other, target_variable):
        """ Evaluate and accumulate loss given outputs and expected results.

        This method is called after each batch with the batch outputs and
        the target (expected) results.  The loss and normalization term are
        accumulated in this method.

        Args:
            decoder_outputs (torch.Tensor): outputs of a batch.
            other (dictionary): extra outputs of the model
            target_variable (torch.Tensor): expected output of a batch.
        """
        targets = other[self.target]
        outputs = other[self.inputs]

        # The input sequence has to be longer than one word in order to calculate a loss
        # In this case targets will have 0 elements, because self.target = shifted_input_variables, i.e. the input
        # sequence without the first element (you can't predict the first token from nothing)
        if len(targets) > 0:
            for step, step_output in enumerate(outputs):
                target = targets[:, step]
                self.eval_step(step_output, target)
        else:
            # TODO: Find a better solution to this
            # When a sequence is too short to calculate a loss, acc_loss will remain a simple float and calling
            # backward() on it produces an error
            dummy_var = autograd.Variable(torch.Tensor([[0]]), requires_grad=True)
            dummy_target = autograd.Variable(torch.LongTensor([0]))
            self.acc_loss += self.criterion(dummy_var, dummy_target)
            self.norm_term += 1


class AnticipationEmbeddingLoss(Loss):
    """
    Loss imposed on an extended encoder that also tries to predict the next word in the sequence,
    but use distance between embeddings as a proxy instead computing a cross-entropy error.
    """
    _NAME = "Anticipation Embedding Loss"
    _SHORTNAME = "antcp_emb_loss"
    _INPUTS = "encoder_predicted_embeddings"
    _TARGETS = "shifted_input_embeddings"

    def __init__(self, size_average=True):
        self.size_average = size_average

        super().__init__(
            self._NAME, self._SHORTNAME, self._INPUTS, self._TARGETS,
            nn.MSELoss( size_average=size_average))

    def get_loss(self):
        if isinstance(self.acc_loss, int):
            return 0
        # total loss for all batches
        loss = self.acc_loss.item()
        if self.size_average:
            # average loss per batch
            loss /= self.norm_term
        return loss

    def eval_step(self, step_outputs, target):
        batch_size = target.size(0)
        outputs = step_outputs.contiguous().view(batch_size, -1)
        self.acc_loss += self.criterion(outputs, target)
        self.norm_term += 1

    def eval_batch(self, decoder_outputs, other, target_variable):
        """ Evaluate and accumulate loss given outputs and expected results.

        This method is called after each batch with the batch outputs and
        the target (expected) results.  The loss and normalization term are
        accumulated in this method.

        Args:
            decoder_outputs (torch.Tensor): outputs of a batch.
            other (dictionary): extra outputs of the model
            target_variable (torch.Tensor): expected output of a batch.
        """
        targets = other[self.target]
        outputs = other[self.inputs]

        # The input sequence has to be longer than one word in order to calculate a loss
        # In this case targets will have 0 elements, because self.target = shifted_input_variables, i.e. the input
        # sequence without the first element (you can't predict the first token from nothing)
        if len(targets) > 0:
            for step, step_output in enumerate(outputs):
                target = targets[:, step]
                self.eval_step(step_output, target)
        else:
            # TODO: Find a better solution to this
            # When a sequence is too short to calculate a loss, acc_loss will remain a simple float and calling
            # backward() on it produces an error
            dummy_var = autograd.Variable(torch.Tensor([0]), requires_grad=True)
            dummy_target = autograd.Variable(torch.Tensor([0]))
            self.acc_loss += self.criterion(dummy_var, dummy_target)
            self.norm_term += 1
