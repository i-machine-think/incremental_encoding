"""
Check whether incrementality metrics defined in incremental_metrics.py are correlated and plot corresponding graphs.
"""

# STD
from itertools import combinations
from collections import defaultdict
from typing import Callable, Optional

# EXT
from machine.trainer import SupervisedTrainer
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

# PROJECT
from test_incrementality import init_argparser, load_test_data, IncrementalEvaluator, METRICS, TOP_N, \
    load_models_from_paths, is_incremental, has_attention
from incremental_models import BottleneckDecoderRNN

# CONST
BASELINE_COLOR = "tab:blue"
INCREMENTAL_COLOR = "tab:orange"
SHORTHANDS = {metric._NAME: short_name for short_name, metric in METRICS.items()}


def test_metric_correlation(measurements: dict):
    """
    Test correlation of metric values for the given models using Pearson's correlation coefficient.
    """
    scores = defaultdict(list)
    correlations = {}

    # Reorder measurements from dict model -> metric -> score to metric -> scores
    for model, metric_scores in measurements.items():
        for metric, score in metric_scores.items():
            scores[metric].append(score)

    # Calculate Pearson's rho
    for metric_a, metric_b in combinations(scores.keys(), 2):
        rho, _ = pearsonr(scores[metric_a], scores[metric_b])
        print(f"{metric_a} | {metric_b}: {rho:.4f}")
        correlations[(metric_a, metric_b)] = rho

    return correlations, scores


def create_correlation_heatmap(correlations: dict, save_path: Optional[str] = None):
    """
    Create a heat map out of all the correlations scores between metrics.
    """
    # Create new dict for all possible correlation pairs
    all_correlations = dict(correlations)

    # Add inverse pairs
    for metric_a, metric_b in correlations.keys():
        all_correlations[(metric_b, metric_a)] = all_correlations[(metric_a, metric_b)] = correlations[(metric_a, metric_b)]

    metrics_a, metrics_b = zip(*correlations.keys())
    metrics = set(metrics_a) | set(metrics_b)

    # Add correlations with itself
    for metric in metrics:
        all_correlations[(metric, metric)] = 1

    correlation_map = np.array([
        [
            all_correlations[(metric_a, metric_b)] if j <= i else 0
            for j, metric_b in enumerate(metrics)
        ] for i, metric_a in enumerate(metrics)
    ])

    fig, ax = plt.subplots()
    img = ax.imshow(correlation_map, cmap="coolwarm", vmin=-1, vmax=1)

    # Create colorbar
    cbar = ax.figure.colorbar(img, ax=ax)
    cbar.ax.set_ylabel("Pearson's rho", rotation=-90, va="bottom")

    # Set ticks
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_yticks(np.arange(len(metrics)))
    ax.set_xticklabels([SHORTHANDS[metric] for metric in metrics])
    ax.set_yticklabels([SHORTHANDS[metric] for metric in metrics])

    # Rotate the tick labels and set their alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations
    for i in range(len(metrics)):
        for j in range(len(metrics)):
            if j > i:
                continue
            ax.text(j, i, round(correlation_map[i, j], 2), ha="center", va="center", color="w", size=13)

    fig.tight_layout()

    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)

    plt.close()


def create_metric_scatter_plot(measurements: dict, image_dir: str, correlations: dict=None,
                               distinction_func: Callable=is_incremental, marker_func: Callable = lambda model: "o",
                               labels_func: Optional[Callable] = None, color_func: Optional[Callable] =None):
    """
    Create scatter plots showing where models fall with respect to two different metrics.
    """
    metrics = list(measurements[list(measurements.keys())[0]].keys())  # Get the name of all used metrics

    for metric_a, metric_b in combinations(metrics, 2):
        # Select two metrics to use as scatter plot axes
        # Now get the model measurements
        xs, ys = defaultdict(list), defaultdict(list)

        for model, scores in measurements.items():
            xs[distinction_func(model)].append(scores[metric_a])
            ys[distinction_func(model)].append(scores[metric_b])

        # Plot points for different models
        for model_type in xs.keys():
            marker = marker_func(model_type)
            color = None if color_func is None else color_func(model_type)
            label = None if labels_func is None else labels_func(model_type)

            plt.scatter(xs[model_type], ys[model_type], c=color, marker=marker, label=label)

        plt.xlabel(metric_a)
        plt.ylabel(metric_b)

        # Plot diagonal that would signify perfect positive (linear) correlation
        ax = plt.gca()
        plt.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".3")
        plt.legend()

        # Display correlation coefficient if available
        if correlations is not None:
            rho = correlations[(metric_a, metric_b)]
            plt.text(0.85, 0.025, r"$\rho={:.2f}$".format(rho), transform=plt.gca().transAxes)

        plt.tight_layout()
        plt.savefig(
            f"{image_dir}/scatter_{metric_a.lower().replace(' ', '_')}_{metric_b.lower().replace(' ', '_')}.png"
        )
        plt.close()


def generate_measurements(models: list, metrics: list):
    """
    Generate measurements for models with respect to given metrics.
    """
    measurements = {}

    # Generate measurements for models
    for model in models:
        evaluator = IncrementalEvaluator(metrics=metrics, batch_size=opt.batch_size)
        metric_results = evaluator.evaluate(model, test, SupervisedTrainer.get_batch_data)
        measurements[model] = {metric._NAME: metric.get_val() for metric in metric_results}

    return measurements


def is_window_bottleneck(model):
    if isinstance(model.decoder_module, BottleneckDecoderRNN):
        if model.decoder_module.bottleneck_type == "window":
            return True

    return False


def get_window_size(model):
    if isinstance(model.decoder_module, BottleneckDecoderRNN):
        if model.decoder_module.bottleneck_type == "window":
            return model.decoder_module.window_size

        return -1


def is_past_bottleneck(model):
    if isinstance(model.decoder_module, BottleneckDecoderRNN):
        if model.decoder_module.bottleneck_type == "past":
            return True

    return False


if __name__ == "__main__":
    parser = init_argparser()
    parser.add_argument("--img_path", help="Path to directory in which to save generated plots.")
    opt = parser.parse_args()

    # Prepare data set
    test, src, tgt = load_test_data(opt)

    # Load models
    models, input_vocab, output_vocab = load_models_from_paths(opt.models, src, tgt)
    pad = output_vocab.stoi[tgt.pad_token]
    metrics = [METRICS[metric](max_len=opt.max_len, pad=pad, n=TOP_N) for metric in opt.metrics]

    # Perform analyses
    measurements = generate_measurements(models, metrics)
    correlations, _ = test_metric_correlation(measurements)

    # Plot heatmap
    create_correlation_heatmap(correlations, save_path=f"{opt.img_path}correlations.png")

    # Plot
    def distinction_function(model):
        if is_incremental(model):
            return "Anticipation"
        elif is_window_bottleneck(model):
            if get_window_size(model) == 1:
                return "Window=1"
            elif get_window_size(model) == 2:
                return "Window=2"
        elif is_past_bottleneck(model):
            return "Past"
        elif has_attention(model):
            return "Attention"
        else:
            return "Baseline"

    def marker_function(model_name):
        markers = {
            "Anticipation": "x",
            "Attention": "o",
            "Baseline": "D",
            "Window=1": "+",
            "Window=2": "+",
            "Past": "*"
        }
        return markers[model_name]

    def color_function(model_name):
        colors = {
            "Anticipation": "tab:red",
            "Attention": "tab:blue",
            "Baseline": "tab:green",
            "Window=1": "tab:orange",
            "Window=2": "tab:purple",
            "Past": "tab:gray"
        }
        return colors[model_name]

    create_metric_scatter_plot(
        measurements, opt.img_path, correlations, distinction_func=distinction_function,
        labels_func=lambda model_name: model_name, color_func=color_function, marker_func=marker_function
    )
