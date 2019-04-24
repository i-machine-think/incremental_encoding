"""
Test the incrementality of a model by computing different scores.
"""

# STD
import argparse
from collections import defaultdict
from typing import Callable

# EXT
from machine.util.checkpoint import Checkpoint
import torch
from machine.trainer import SupervisedTrainer
from machine.dataset import SourceField, TargetField
import torchtext
from machine.evaluator import Evaluator
import numpy as np
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt

# PROJECT
from incremental_metrics import AverageIntegrationRatio, DiagnosticClassifierAccuracy, \
    WeighedDiagnosticClassifierAccuracy, RepresentationalSimilarity, SequenceAccuracyWrapper, WordAccuracyWrapper

# GLOBALS
from incremental_models import IncrementalSeq2Seq

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
METRICS = {
    "int_ratio": AverageIntegrationRatio,
    "dc_acc": DiagnosticClassifierAccuracy,
    "wdc_acc": WeighedDiagnosticClassifierAccuracy,
    "repr_sim": RepresentationalSimilarity,
    "seq_acc": SequenceAccuracyWrapper,
    "word_acc": WordAccuracyWrapper
}

# CONSTANTS
TOP_N = 3


def is_incremental(model):
    """
    Test whether a model is of an incremental model class.
    """
    return isinstance(model, IncrementalSeq2Seq)


def has_attention(model):
    """
    Test whether a model uses attention.
    """
    return model.decoder_module.attention_method is not None


def test_incrementality(distinction_func: Callable=is_incremental, model_names: tuple=None):
    parser = init_argparser()

    opt = parser.parse_args()

    if torch.cuda.is_available():
        print("Cuda device set to %i" % opt.cuda_device)
        torch.cuda.set_device(opt.cuda_device)

    parser = init_argparser()
    opt = parser.parse_args()

    # Prepare data set
    test, src, tgt = load_test_data(opt)

    # Load models
    models, input_vocab, output_vocab = load_models_from_paths(opt.models, src, tgt)
    pad = output_vocab.stoi[tgt.pad_token]

    metrics = [METRICS[metric](max_len=opt.max_len, pad=pad, n=TOP_N) for metric in opt.metrics]

    # Evaluate models on test set
    first_measurements = defaultdict(list)
    second_measurements = defaultdict(list)

    for model, model_path in zip(models, opt.models):
        evaluator = IncrementalEvaluator(metrics=metrics, batch_size=1)
        metrics = evaluator.evaluate(model, test, SupervisedTrainer.get_batch_data)

        print(type(model).__name__)
        print(model_path)
        print(
            "\n".join([f"{metric._NAME:<40}: {metric.get_val():4f}" for metric in metrics])
        )
        print("")

        for metric in metrics:
            if distinction_func(model):
                second_measurements[metric._SHORTNAME].append(metric.get_val())
            else:
                first_measurements[metric._SHORTNAME].append(metric.get_val())

    # Evaluate all the models together and generate final report
    print("\nResults per metric")
    for metric in first_measurements.keys():
        first_results, second_results = np.array(first_measurements[metric]), np.array(second_measurements[metric])
        first_avg, first_std = first_results.mean(), first_results.std()
        second_avg, second_std = second_results.mean(), second_results.std()
        _, p_value = ttest_ind(first_results, second_results, equal_var=False)
        first_name, second_name = ("Baseline", "Attention") if model_names is None else model_names

        print(
            f"{metric:<10}: {first_name} {first_avg:.4f} ±{first_std:.3f} | {second_name} {second_avg:.4f} "
            f"±{second_std:.3f} | p={p_value:.4f}"
        )

    # Plot as bar graph
    create_results_bar_plot(
        first_measurements, second_measurements, names=("Baseline", "Attention"), colors=("tab:blue", "tab:orange"),
        img_path="./img/bars.png"
    )


def create_results_bar_plot(first_measurements, second_measurements, names, colors, img_path=None):
    """
    Create bar plot with whiskers to show the metric results for two distinct groups of models.
    """
    metrics = first_measurements.keys()
    num_metrics = len(metrics)
    fig, ax = plt.subplots()
    width = 0.3
    ind = np.arange(num_metrics)

    first_avgs, first_stds = [None] * num_metrics, [None] * num_metrics
    second_avgs, second_stds = [None] * num_metrics, [None] * num_metrics

    for i, metric in enumerate(metrics):
        first_results, second_results = np.array(first_measurements[metric]), np.array(second_measurements[metric])
        first_avgs[i], first_stds[i] = first_results.mean(), first_results.std()
        second_avgs[i], second_stds[i] = second_results.mean(), second_results.std()

    p1 = ax.bar(ind, first_avgs, width, color=colors[0], yerr=first_stds)
    p2 = ax.bar(ind + width, second_avgs, width, color=colors[1], yerr=second_stds)

    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(metrics)

    ax.legend((p1[0], p2[0]), names)
    ax.autoscale_view()

    if img_path is None:
        plt.show()
    else:
        plt.tight_layout()
        plt.savefig(img_path)


class IncrementalEvaluator(Evaluator):
    def update_batch_metrics(self, metrics, other, target_variable):
        """
        Update a list with metrics for current batch.

        Args:
            metrics (list): list with of machine.metric.Metric objects
            other (dict): dict generated by forward pass of model to be evaluated
            target_variable (dict): map of keys to different targets of model

        Returns:
            metrics (list): list with updated metrics
        """
        # evaluate output symbols
        outputs = other['sequence']

        for metric in metrics:
            metric.eval_batch(outputs, other)

        return metrics

    def evaluate(self, model, data, get_batch_data):
        """ Evaluate a model on given dataset and return performance.

        Args:
            model (machine.models): model to evaluate
            data (machine.dataset.dataset.Dataset): dataset to evaluate against

        Returns:
            loss (float): loss of the given model on the given dataset
            accuracy (float): accuracy of the given model on the given dataset
        """
        # If the model was in train mode before this method was called, we make sure it still is
        # after this method.
        previous_train_mode = model.training
        model.eval()

        metrics = self.metrics
        for metric in metrics:
            metric.reset()

        # create batch iterator
        batch_iterator = torchtext.data.BucketIterator(
            dataset=data, batch_size=self.batch_size,
            sort=True, sort_key=lambda x: len(x.src),
            device=device, train=False
        )

        # loop over batches
        with torch.no_grad():
            for batch in batch_iterator:
                input_variable, input_lengths, target_variable = get_batch_data(batch)

                decoder_outputs, decoder_hidden, other = model(input_variable, input_lengths.tolist(), target_variable)

                # Get other necessary information for eval
                other["input_sequences"] = input_variable
                encoder_results = model.encoder_module(input_variable, input_lengths)
                other["encoder_hidden"] = encoder_results[0]
                other["encoder_embeddings"] = model.encoder_module.embedding(input_variable)
                other["decoder_output"] = target_variable["decoder_output"]  # Store everything used for eval in other
                other["encoder"] = model.encoder_module.rnn

                # Compute metric(s) over one batch
                metrics = self.update_batch_metrics(metrics, other, target_variable)

        model.train(previous_train_mode)

        return metrics

    def evaluate_stepwise(self, model, data, get_batch_data, break_after):
        # If the model was in train mode before this method was called, we make sure it still is
        # after this method.
        previous_train_mode = model.training
        model.encoder_module.variable_lengths = False
        model.eval()

        # create batch iterator
        batch_iterator = torchtext.data.BucketIterator(
            dataset=data, batch_size=self.batch_size,
            sort=False, sort_key=lambda x: len(x.src),
            device=device, train=False
        )

        metric_results = []

        metrics = self.metrics
        for metric in metrics:
            metric.reset()

        # loop over batches
        with torch.no_grad():
            for i, batch in enumerate(batch_iterator):

                if i > break_after:
                    break

                batch_results = []
                input_variable, input_lengths, target_variable = get_batch_data(batch)

                decoder_outputs, decoder_hidden, other = model(input_variable, input_lengths.tolist(), target_variable)

                # Get other necessary information for eval
                other["input_sequences"] = input_variable
                encoder_results = model.encoder_module(input_variable, input_lengths)
                other["encoder_hidden"] = encoder_results[0]
                other["encoder_embeddings"] = model.encoder_module.embedding(input_variable)
                other["decoder_output"] = target_variable["decoder_output"]  # Store everything used for eval in other
                other["encoder"] = model.encoder_module.rnn

                # Compute metric(s) over one batch
                metrics = self.update_batch_metrics(metrics, other, target_variable)
                metric_results.append([metric.get_val() for metric in metrics])

        model.train(previous_train_mode)

        return metric_results


def load_test_data(opt):
    src = SourceField()
    tgt = TargetField()
    tabular_data_fields = [('src', src), ('tgt', tgt)]

    max_len = opt.max_len

    def len_filter(example):
        return len(example.src) <= max_len and len(example.tgt) <= max_len

    # generate training and testing data
    test = torchtext.data.TabularDataset(
        path=opt.test, format='tsv',
        fields=tabular_data_fields,
        filter_pred=len_filter
    )

    return test, src, tgt


def load_models_from_paths(paths: list, src, tgt):
    """
    Load all the models specified in a list of paths.
    """
    models = []

    for path in paths:
        checkpoint = Checkpoint.load(path)
        models.append(checkpoint.model)

    # Build vocab once
    input_vocab = checkpoint.input_vocab
    src.vocab = input_vocab
    input_vocab = checkpoint.input_vocab
    src.vocab = input_vocab
    output_vocab = checkpoint.output_vocab
    tgt.vocab = output_vocab
    tgt.eos_id = tgt.vocab.stoi[tgt.SYM_EOS]
    tgt.sos_id = tgt.vocab.stoi[tgt.SYM_SOS]

    return models, input_vocab, output_vocab


def init_argparser():
    parser = argparse.ArgumentParser()

    # Model arguments
    parser.add_argument('--test', help='Testing data')
    parser.add_argument('--metrics', nargs='+', default=['seq_acc'],
                        choices=["int_ratio", "dc_acc", "wdc_acc", "repr_sim", "seq_acc", "word_acc"],
                        help='Metrics to use')
    parser.add_argument('--batch_size', type=int,
                        help='Batch size', default=1)
    # Data management
    parser.add_argument('--cuda_device', default=0,
                        type=int, help='set cuda device to use')
    parser.add_argument('--max_len', type=int,
                        help='Maximum sequence length', default=50)
    parser.add_argument("--models", nargs="+", help="List of paths to models used to conduct analyses.")

    return parser


if __name__ == "__main__":
    test_incrementality(distinction_func=has_attention, model_names=("Baseline", "Attention"))
