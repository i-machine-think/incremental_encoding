"""
Implementing metrics in order to measure the degree to which a model processes inputs incrementally.
"""

# STD
import random
import math
from collections import defaultdict

# EXT
from machine.metrics.metrics import Metric, SequenceAccuracy, WordAccuracy
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
import numpy as np
from scipy.stats import itemfreq
import numpy.linalg as linalg


class SequenceAccuracyWrapper(SequenceAccuracy):
    """
    Wrapper for sequence accuracy class in order to make it compatible with the way incremental metrics are used.
    """
    def __init__(self, pad, **kwargs):
        super().__init__(ignore_index=pad)


class WordAccuracyWrapper(WordAccuracy):
    """
    Wrapper for word accuracy class in order to make it compatible with the way incremental metrics are used.
    """
    def __init__(self, pad, **kwargs):
        super().__init__(ignore_index=pad)


class AverageIntegrationRatio(Metric):
    """
    This ratio measure how much of the current encoder hidden state is based on the current input token
    and on the last hidden state.

    For any hidden state h_t and input embedding x_t, the ratio is computed by

    min(||h_t - x_t|| / ||h_t - h_t-1||, ||h_t - h_t-1|| / ||h_t - x_t||)

    which is then averaged over T-1 time steps. To apply this measure, the dimensionality
    has to be the same for hidden states and word embeddings alike.
    """
    _NAME = "Integration ratio"
    _SHORTNAME = "intratio"
    _INPUT = "seqlist"

    def __init__(self, **kwargs):
        super().__init__(self._NAME, self._SHORTNAME, self._INPUT)
        self.batch_ratio = 0

    def get_val(self):
        return self.batch_ratio.cpu().numpy()

    def reset(self):
        self.batch_ratio = 0

    def eval_batch(self, outputs, targets):
        hidden = targets["encoder_hidden"]  # Input embeddings
        embeddings = targets["encoder_embeddings"]  # Hidden states
        batch_size, timesteps, hidden_dim = hidden.size()
        embedding_dim = embeddings.size(2)
        ratios = torch.zeros(batch_size, timesteps - 1)
        encoder_cell = targets["encoder"]

        # Check if dimensionality corresponds for embeddings and hidden states
        assert embedding_dim == hidden_dim, "This metric only works if input embeddings and hidden states are of the" \
                                            "same dimensionality"

        # Calculate ratios
        for t in range(1, timesteps):
            h_t, h_previous, x_t = hidden[:, t], hidden[:, t - 1], embeddings[:, t]

            # Do two forward passes: One where the input is ignored and one where the history is ignored
            empty_x = torch.zeros(*x_t.size()).unsqueeze(1)
            compare_x, _ = encoder_cell.forward(x_t.unsqueeze(1))  # If all history was erased
            hx = h_previous.unsqueeze(0)
            _, (compare_h, _) = encoder_cell.forward(empty_x, (hx, hx))  # If input was ignored

            # Compare
            compare_x = compare_x.squeeze(1)
            compare_h = compare_h.squeeze(0)
            delta_x = F.pairwise_distance(h_t, compare_x, p=2, keepdim=True)
            delta_h = F.pairwise_distance(h_t, compare_h, p=2, keepdim=True)

            compared = torch.cat((delta_x / delta_h, delta_h / delta_x), dim=1)
            minimum_score, _ = torch.min(compared, dim=1)
            ratios[:, t - 1] = minimum_score

        self.batch_ratio = ratios.view(batch_size * (timesteps - 1)).sum() / (batch_size * timesteps)


class ActivationsDataset:
    """
    Class to store activations and tokens alike and provide a flexible way to generate datasets for Diagnostic
    Classifiers.
    """

    def __init__(self, max_len, pad, **kwargs):
        self.max_allowed_len = max_len
        self.max_found_len = 0
        self.pad = pad
        self.sentence_activations = None
        self.sentence_tokens = np.empty((0, max_len))
        self.hidden_dim = -1
        self._targets = None

    def add_batch(self, activations: torch.Tensor, tokens: torch.Tensor):
        """
        Add a batch of activation and tokens to the data set and convert it to numpy.
        """
        # Infer hidden state dimensionality from first sample
        if self.sentence_activations is None:
            _, _, self.hidden_dim = activations.size()
            self.sentence_activations = np.empty((0, self.max_allowed_len, self.hidden_dim))

        batch_size, num_activations, _ = activations.size()
        if num_activations > self.max_found_len:
            self.max_found_len = num_activations

        # Add activations
        new_activations = np.zeros((batch_size, self.max_allowed_len, self.hidden_dim)) + self.pad
        activations = activations.cpu().numpy()
        new_activations[:, :activations.shape[1], :] = activations
        self.sentence_activations = np.concatenate((self.sentence_activations, new_activations), axis=0)

        # Add tokens
        new_tokens = np.zeros((batch_size, self.max_allowed_len)) + self.pad
        tokens = tokens.cpu().numpy()
        new_tokens[:, :tokens.shape[1]] = tokens
        self.sentence_tokens = np.concatenate((self.sentence_tokens, new_tokens), axis=0)

    def select_targets(self, selection_func=None, **selection_kwargs):
        """
        Select the target tokens to be selected at each time step.
        """
        targets = []
        selection_func = selection_func if selection_func is not None else self.select_by_freq

        for t in range(self.max_allowed_len):
            targets.append(selection_func(self.sentence_tokens[:, t], **selection_kwargs))

        return targets

    def select_by_freq(self, target_column, n=5):
        """
        Select Diagnostic classifier targets by frequency (ignoring the padding token).
        """
        target_column = target_column[target_column != self.pad]  # Don't count padding tokens
        target_freqs = itemfreq(target_column)
        target_freqs.sort(axis=1)
        target_freqs = target_freqs[::-1]  # Sort in descending order
        targets = target_freqs[:n, 0]  # Only pick n most common targets

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

        positive_label_weight = np.sum(y.astype(int)) / y.shape[0]
        class_weights = {0: 1 - positive_label_weight, 1: positive_label_weight}

        train_indices, test_indices = self._split_set(X)
        X_train, y_train = X[train_indices, :], y[train_indices]
        X_test, y_test = X[test_indices, :], y[test_indices]

        return X_train, y_train, X_test, y_test, class_weights

    def get_target_activations_at_timestep(self, t, target):
        """
        Select the activations of a target token for a specific time step inside the data set.

        :param t: Time step which activations should be used.
        :param target: Target which activations should be used.
        :return: Numpy array of targets at time step found times activation dimensionality
        """
        activations = self.sentence_activations[:, t]  # All activations at time step t
        tokens = self.sentence_tokens[:, t]  # All tokens at time step t
        target_tokens = tokens == target  # Places where the token is the target token
        selected_activations = activations[target_tokens, :]  # Activations of target token at time step t

        return selected_activations

    @staticmethod
    def _split_set(data, ratio=(0.9, 0.1)):

        if not sum(ratio) == 1:
            raise ValueError('Ratios must sum to 1!')

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

    def __init__(self, max_len, pad, selection_func=None, **selection_kwargs):
        super().__init__(self._NAME, self._SHORTNAME, self._INPUT)
        self.classifiers_trained = False  # Check whether classifiers have already been trained
        self.max_len = max_len
        self.pad = pad
        self.dataset = ActivationsDataset(max_len, pad)
        self.selection_func = selection_func
        self.selection_kwargs = selection_kwargs
        self.classifiers = {}
        self.accuracies = {}

    def eval_batch(self, outputs, targets):
        # Only add activations to the data set here
        hidden = targets["encoder_hidden"]
        tokens = targets["input_sequences"]
        self.dataset.add_batch(hidden, tokens)

    def get_val(self):
        # Only train classifiers once
        if not self.classifiers_trained:
            self.train_classifiers()

        acc = sum([
            self.weight(t, t_prime, target) * accuracy
            for (t, t_prime, target), accuracy in self.accuracies.items()
        ])
        acc *= self.norm_factor

        return acc

    def reset(self):
        self.classifiers_trained = False
        self.dataset = ActivationsDataset(self.max_len, self.pad)
        self.classifiers = {}
        self.accuracies = {}

    def train_classifiers(self):
        # Generate target tokens to predict
        targets_per_timestep = self.dataset.select_targets(self.selection_func)

        # Train one classifier per target per time step to the end of the sequence
        T = self.dataset.max_found_len
        num_classifiers = sum([
            (T - t) * len(targets)
            for t, targets in zip(range(1, T), targets_per_timestep[:T])
        ])
        current_classifier = 1

        # Now train a whole lot of classifiers and store their accuracies
        # Skip first time step as there wouldn't be anything to predict
        for t in range(1, T):
            for t_prime, targets in zip(range(0, t), targets_per_timestep[:t]):
                for target in targets:
                    print(
                        f"\rTrained {current_classifier}/{num_classifiers} Diagnostic classifiers...",
                        end="", flush=True
                    )

                    # Train Diagnostic Classifier
                    X_train, y_train, X_test, y_test, class_weights = self.dataset.generate_training_data(
                        t, t_prime, target
                    )
                    dg = LogisticRegression(class_weight=class_weights, solver="lbfgs", max_iter=200)
                    dg.fit(X_train, y_train)

                    # Test the whole shebang
                    pred = dg.predict(X_test)
                    acc = np.sum(pred == y_test) / y_test.shape[0]

                    # Save everything
                    key = (t, t_prime, target)
                    self.classifiers[key], self.accuracies[key] = dg, acc

                    current_classifier += 1

        self.classifiers_trained = True
        print("")

    @staticmethod
    def weight(*args):
        return 1

    @property
    def norm_factor(self):
        return 1 / len(self.classifiers)


class WeighedDiagnosticClassifierAccuracy(DiagnosticClassifierAccuracy):
    """
    Same as DiagnosticClassifierAccuracy, but the accuracies are weighed by the distance between the time step of the
    hidden activations used as input and the time step of the token to predict.
    """
    _NAME = "Weighed Diagnostic Classifier Accuracy"
    _SHORTNAME = "wdc_acc"
    _INPUT = "seqlist"

    @staticmethod
    def weight(t, t_prime, *args):
        return t - t_prime

    @property
    def norm_factor(self):
        T = self.dataset.max_found_len
        targets_per_timestep = self.dataset.select_targets()

        norm = 0

        # Normalize by sum of distances between the time step of which the activations are used and the time step of the
        # target to be predicted; also consider how many different targets are weighed by this factor
        for t in range(1, T):
            for t_prime, targets in zip(range(0, t), targets_per_timestep[:t]):
                norm += (t - t_prime) * len(targets)

        return 1 / norm


class RepresentationalSimilarity(Metric):
    """
    A metric expressing the similarity between hidden states concerning the same token during processing.
    We would expect the hidden states to be more similar for an incremental model after processing the same token after
    an arbitrary prefix than for the baseline model. E.g. consider the sequences

    t1 t2 t3
    --------
    XX T1 XX
    XX XX XX

    where T1 is a specific token and XX are arbitrary tokens. Here we would expect the representations for sequence 1
    to be more similar after encoding t3 for a model with incremental capabilities.
    We can quantify this difference by calculating the average euclidean distance over all hidden representations
    encoded at t1.

    In essence, the final score expresses the distance between activations produced after processing the same input
    token at the same time step, averaged over all target tokens selected over all time steps in the data set (selected
    tokens depend on the selection function, default is all n most frequent tokens at that time step).
    """
    _NAME = "Representational Similarity"
    _SHORTNAME = "repsim"
    _INPUT = "seqlist"

    def __init__(self, max_len, pad, selection_func=None, **selection_kwargs):
        super().__init__(self._NAME, self._SHORTNAME, self._INPUT)
        self.max_len = max_len
        self.pad = pad
        self.selection_func = selection_func
        self.selection_kwargs = selection_kwargs
        self.dataset = ActivationsDataset(max_len, pad)
        self.similarities_calculated = False  # Check whether similarities have already been calculated
        self.average_distances = [defaultdict(float) for _ in range(max_len)]
        self.representational_sim = 0  # Average distance of all targets of all time steps

    def eval_batch(self, outputs, targets):
        # Only add activations to the data set here
        hidden = targets["encoder_hidden"]
        tokens = targets["input_sequences"]
        self.dataset.add_batch(hidden, tokens)

    def get_val(self):
        if not self.similarities_calculated:
            self.calculate_similarities()

        return self.representational_sim

    def reset(self):
        self.similarities_calculated = False
        self.dataset = ActivationsDataset(self.max_len, self.pad)
        self.average_distances = [defaultdict(float) for _ in range(self.max_len)]

    def calculate_similarities(self):
        # Generate target tokens calculate similarities for
        targets_per_timestep = self.dataset.select_targets(self.selection_func)
        T = self.dataset.max_found_len
        global_norm = sum([len(targets) for targets in targets_per_timestep])
        global_cumulative_distances = 0

        # Calculate average distances
        for t in range(T):
            for target in targets_per_timestep[t]:
                # Activations for target tokens at time step t that are found in data set.
                target_activations = self.dataset.get_target_activations_at_timestep(t, target)
                average_distance = self.calculate_average_distance(target_activations)
                self.average_distances[t][target] = average_distance
                global_cumulative_distances += average_distance

        self.representational_sim = global_cumulative_distances / global_norm

    @staticmethod
    def calculate_average_distance(activations):
        """
        Calculate the average euclidean distance between activations.
        """
        num_activations = activations.shape[0]
        norm_factor = num_activations * (num_activations - 1) / 2  # Number of comparisons to make
        cumulative_distance = 0

        # Compute distances between pairs of activations
        # Do not consider: Distance of activation to itself and for a pair that has been computed earlier
        for i in range(num_activations - 1):
            for j in range(i + 1, num_activations):
                distance = linalg.norm(activations[i, :] - activations[j, :])
                cumulative_distance += distance

        return cumulative_distance / norm_factor
