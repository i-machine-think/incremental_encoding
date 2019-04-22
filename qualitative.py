"""
Perform some qualitative analyses by showing where models diverge on their metric scores on sentence samples.
"""

# STD
import argparse
from collections import namedtuple

# EXT
import numpy as np
import torch
from machine.trainer import SupervisedTrainer
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.cm import ScalarMappable

# PROJECT
from test_incrementality import (
    IncrementalEvaluator, load_models_from_paths, load_test_data, has_attention, METRICS, TOP_N
)

# TYPES
ScoredSentence = namedtuple("ScoredSentence", ["src", "tgt", "first_scores", "second_scores", "diff"])


def qualitative_analysis(samples, img_dir, distinction_func=has_attention, model_names=None, offset=0):
    parser = init_argparser()

    opt = parser.parse_args()

    if torch.cuda.is_available():
        print("Cuda device set to %i" % opt.cuda_device)
        torch.cuda.set_device(opt.cuda_device)

    parser = init_argparser()
    opt = parser.parse_args()

    # Prepare data set
    test, src, tgt = load_test_data(opt)
    break_after = opt.break_after if opt.break_after is not None else len(test)

    # Load models
    (first_model, second_model), input_vocab, output_vocab = load_models_from_paths(
        [opt.first_model, opt.second_model], src, tgt
    )
    pad = output_vocab.stoi[tgt.pad_token]

    metrics = [METRICS[metric](max_len=opt.max_len, pad=pad, top_n=TOP_N, reduce=False) for metric in opt.metrics]

    # TODO: Support multiple metrics at once
    assert len(metrics) == 1, "Only one metric at a time supported!"
    assert opt.batch_size == 1, "Not using batch_size = 1 distorts results"

    # Collect scores on dataset
    evaluator = IncrementalEvaluator(metrics=metrics, batch_size=opt.batch_size)
    print("Fetch results for first model...")
    first_scores = evaluator.evaluate_stepwise(first_model, test, SupervisedTrainer.get_batch_data, break_after)
    print("Fetch results for second model...")
    second_scores = evaluator.evaluate_stepwise(second_model, test, SupervisedTrainer.get_batch_data, break_after)

    # Aggregate scores
    scored_sentences = []
    for sentence, first_score, second_score in zip(test.examples, first_scores, second_scores):
        # TODO: Support multiple metrics at once
        first_score, second_score = first_score[0], second_score[0]

        scored_sentence = ScoredSentence(
            src=sentence.src, tgt=sentence.tgt, first_scores=first_score, second_scores=second_score,
            diff=np.mean(np.abs(first_score-second_score))
        )
        scored_sentences.append(scored_sentence)

    # Sort sentences descending by average different score assigned to individual time steps by the given metric
    sorted_scored_sentences = sorted(scored_sentences, key=lambda sen: sen.diff, reverse=True)

    # Plot most different samples
    for i, scored_sentence in enumerate(sorted_scored_sentences[:samples]):
        plot_stepwise_metric(scored_sentence, model_names, img_dir, opt, i)


def plot_stepwise_metric(scored_sentence, model_names, img_dir, opt, num):
    tokens = scored_sentence.src
    # TODO: Make this metric agnostic
    first_scores = np.concatenate(([[1]], scored_sentence.first_scores), axis=1).squeeze(0)
    second_scores = np.concatenate(([[1]], scored_sentence.second_scores), axis=1).squeeze(0)
    fig, ax = plt.subplots()
    img = ax.imshow([first_scores, second_scores], cmap="seismic", vmin=0.7, vmax=1.3)

    # We want to show all ticks...
    ax.set_xticks(np.arange(len(tokens)))
    ax.set_yticks(np.arange(2))
    # ... and label them with the respective list entries
    ax.set_xticklabels(tokens)
    ax.set_yticklabels(model_names)

    ax.text(0, 0, "X", ha="center", va="center", color="black", fontsize=20)
    ax.text(0, 1, "X", ha="center", va="center", color="black", fontsize=20)
    for i in range(1, len(tokens)):
        ax.text(i, 0, round(first_scores[i], 4), ha="center", va="center", color="w")
        ax.text(i, 1, round(second_scores[i], 4), ha="center", va="center", color="w")

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    cbar = ax.figure.colorbar(img, ax=ax, shrink=0.4)
    cbar.ax.set_ylabel("Metric score", rotation=-90, va="bottom")

    plt.tight_layout()
    plt.savefig(f"{img_dir}/{opt.metrics[0]}_{num}.png")
    plt.close()


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
    parser.add_argument("--first_model", type=str)
    parser.add_argument("--second_model", type=str)
    parser.add_argument("--break_after", type=int)

    return parser


if __name__ == "__main__":
    qualitative_analysis(
        model_names=("Baseline", "Attention"), samples=15, img_dir="./img", offset=1
    )