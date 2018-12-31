"""
Check whether incrementality metrics defined in incremental_metrics.py are correlated and plot corresponding graphs.
"""

# STD
from itertools import combinations
from collections import defaultdict

# EXT
from machine.util.checkpoint import Checkpoint
from machine.trainer import SupervisedTrainer
from scipy.stats import pearsonr

# PROJECT
from test_incrementality import init_argparser, load_test_data, IncrementalEvaluator, METRICS, TOP_N


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

    return correlations


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


def load_models_from_paths(paths: list):
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


if __name__ == "__main__":
    parser = init_argparser()
    parser.add_argument("--models", nargs="+", help="List of paths to models used to conduct analyses.")
    opt = parser.parse_args()

    # Prepare data set
    test, src, tgt = load_test_data(opt)

    # Load models
    models, input_vocab, output_vocab = load_models_from_paths(opt.models)
    pad = output_vocab.stoi[tgt.pad_token]
    metrics = [METRICS[metric](max_len=opt.max_len, pad=pad, n=TOP_N) for metric in opt.metrics]

    # Perform analyses
    measurements = generate_measurements(models, metrics)
    test_metric_correlation(measurements)


