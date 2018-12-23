"""
Implementing metrics in order to measure the degree to which a model processes inputs incrementally.
"""

# STD
import random
import math

# EXT
from machine.metrics.metrics import Metric
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
import numpy as np
from scipy.stats import itemfreq


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


class ActivationsDataset:
    """
    Class to store activations and tokens alike and provide a flexible way to generate datasets for Diagnostic
    Classifiers.
    """

    def __init__(self, max_len, pad):
        self.max_len = max_len
        self.pad = pad
        self.sentence_activations = np.empty((0, max_len))
        self.sentence_tokens = np.empty((0, max_len))

    def add_batch(self, activations: torch.Tensor, tokens: torch.Tensor):
        """
        Add a batch of activation and tokens to the data set and convert it to numpy.
        """
        self.sentence_activations.concatenate(activations.cpu().numpy(), axis=0)
        self.sentence_tokens.concatenate(tokens.cpu().numpy(), axis=0)

    def select_targets(self, selection_func=None):
        """
        Select the target tokens to be selected at each time step.
        """
        targets = []
        selection_func = selection_func if selection_func is not None else self.select_by_freq

        for t in range(self.max_len):
            targets.append(selection_func(self.sentence_tokens[:, t]))

        return targets

    def select_by_freq(self, target_column, n=5):
        """
        Select Diagnostic classifier targets by frequency (ignoring the padding token).
        """
        target_column = target_column.delete(target_column == self.pad)  # Don't count padding tokens
        target_freqs = itemfreq(target_column)
        target_freqs = target_freqs.sort(axis=1)[::-1]
        targets = target_freqs[:, :n]  # Only pick n most common targets

        return targets

    def generate_training_data(self, t, t_prime, target):
        """
        Generate the training data for diagnostic classifier at time step t predicting a target at time step t_prime.

        :param t: Time step whose activations the DG is using a input.
        :param t_prime: Time step for which the token is supposed to be predicted.
        :return: Tuple of training and testing data X, labels y and class weights.
        """
        X = self.sentence_activations[:, t]  # All activations at time step t
        y_values = self.sentence_tokens[:, t_prime]  # All tokens at time step t_prime
        y = y_values == target  # 1 where the token in question is the target token

        positive_label_weight = np.sum(y) / y.shape[0]
        class_weights = {0: 1 - positive_label_weight, 1: positive_label_weight}

        train_indices, test_indices = self._split_set(X)
        X_train, y_train = X[train_indices, :]
        X_test, y_test = X[test_indices, :]

        return X_train, y_train, X_test, y_test, class_weights

    @staticmethod
    def _split_set(data, ratio=(0.9, 0.1)):

        if not sum(ratio) == 1: raise ValueError('Ratios must sum to 1!')
        length = data.shape[0]
        train_cutoff = math.floor(length * ratio[0])

        indices = list(range(length))
        random.shuffle(indices)
        train_indices = indices[:train_cutoff]
        test_indices = indices[train_cutoff:]

        return train_indices, test_indices


class DiagnosticClassifierAccuracy(Metric):
    """
    Calculate the average accuracy of diagnostic classifiers trying to predict the existence of a token during a
    specific time step based on the hidden activation of a later time step.

    Because we want to utilize all the testing data to train Diagnostic Classifiers, eval_batch() is only used to gather
    and structure the training data; the classifiers themselves are trained only once get_val() is called, therefore
    this step can take a bit longer.
    """
    _NAME = "Diagnostic Classifier Accuracy"
    _SHORTNAME = "dc_acc"
    _INPUT = "seqlist"

    def __init__(self, max_len, pad, selection_func=None):
        super().__init__(self._NAME, self._SHORTNAME, self._INPUT)
        self.classifiers_trained = False  # Check whether classifiers have already been trained
        self.max_len = max_len
        self.pad = pad
        self.dataset = ActivationsDataset(max_len, pad)
        self.selection_func = selection_func
        self.classifiers = {}
        self.accuracies = {}

    def eval_batch(self, outputs, targets):
        # Only add activations to the data set here
        hidden = targets["encoder_hidden"]
        self.dataset.add(hidden, targets)

    def get_val(self):
        ... # TODO

    def reset(self):
        self.classifiers_trained = False
        self.dataset = ActivationsDataset(self.max_len, self.pad)

    def train_classifiers(self):
        # Generate target tokens to predict
        targets_per_timestep = self.dataset.select_targets(self.selection_func)

        # Now train a whole lot of classifiers and store their accuracies
        # Skip first time step as there wouldn't be anything to predict
        for t in range(1, self.max_len):
            for t_prime, targets in zip(range(0, t), targets_per_timestep[:t]):
                for target in targets:
                    # Train
                    X_train, y_train, X_test, y_test, class_weights = self.dataset.generate_training_data(t, t_prime, target)
                    dg = LogisticRegression(class_weight=class_weights)
                    dg.fit(X_train, y_train)

                    # Test the whole shebang
                    pred = dg.predict(X_test)
                    acc = np.sum(pred == y_test) / y_test.shape[0]

                    # Save everything
                    key = (t, t_prime, target)
                    self.classifiers[key] = dg, self.accuracies[key] = acc



    def weight(self, **args):
        ...  # TODO

    @@property
    def norm_factor(self):
        ...  # TODO


class WeighedDiagnosticClassifierAccuracy(DiagnosticClassifierAccuracy):
    """
    Same as DiagnosticClassifierAccuracy, but the accuracies are weighed by the distance between the time step of the
    hidden activations used as input and the time step of the token to predict.
    """
    ...  # TODO




