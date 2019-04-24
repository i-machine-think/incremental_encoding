"""
Perform some qualitative analyses by showing where models diverge on their metric scores on sentence samples.
"""

# STD
import argparse
from collections import namedtuple
import random

# EXT
import numpy as np
import torch
from machine.trainer import SupervisedTrainer
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf

# PROJECT
from test_incrementality import (
    IncrementalEvaluator, load_models_from_paths, load_test_data, has_attention, METRICS, TOP_N
)

# TYPES
ScoredSentence = namedtuple(
    "ScoredSentence", ["src", "tgt", "first_scores", "first_std", "second_scores", "second_std", "diff"]
)


def qualitative_analysis(samples, img_dir, distinction_func=has_attention, model_names=None, pdf_path=None):
    parser = init_argparser()

    pdf = matplotlib.backends.backend_pdf.PdfPages(pdf_path) if pdf_path is not None else None

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
    models, input_vocab, output_vocab = load_models_from_paths(opt.models, src, tgt)
    first_models, second_models = [], []

    for model in models:
        if has_attention(model):
            second_models.append(model)
        else:
            first_models.append(model)

    pad = output_vocab.stoi[tgt.pad_token]

    metrics = [METRICS[metric](max_len=opt.max_len, pad=pad, top_n=TOP_N, reduce=False) for metric in opt.metrics]

    # TODO: Support multiple metrics at once
    assert len(metrics) == 1, "Only one metric at a time supported!"
    assert opt.batch_size == 1, "Not using batch_size = 1 distorts results"

    # Collect scores on dataset
    evaluator = IncrementalEvaluator(metrics=metrics, batch_size=opt.batch_size)
    print("Fetch results for baseline models...")
    first_scores = []
    for first_model in first_models:
        model_scores = evaluator.evaluate_stepwise(first_model, test, SupervisedTrainer.get_batch_data, break_after)
        first_scores.append(model_scores)

    print("Fetch results for attention models...")
    second_scores = []
    for second_model in second_models:
        model_scores = evaluator.evaluate_stepwise(second_model, test, SupervisedTrainer.get_batch_data, break_after)
        second_scores.append(model_scores)

    # Aggregate scores
    first_scores = [np.concatenate([sc[0] for sc in scores], axis=0) for scores in zip(*first_scores)]
    second_scores = [np.concatenate([sc[0] for sc in scores], axis=0) for scores in zip(*second_scores)]

    scored_sentences = []
    for sentence, first_scores, second_scores in zip(test.examples, first_scores, second_scores):
        first_score, first_std = first_scores.mean(axis=0, keepdims=True), first_scores.std(axis=0, keepdims=True)
        second_score, second_std = second_scores.mean(axis=0, keepdims=True), second_scores.std(axis=0, keepdims=True)

        scored_sentence = ScoredSentence(
            src=sentence.src, tgt=sentence.tgt, first_scores=first_score, second_scores=second_score,
            first_std=first_std, second_std=second_std,
            diff=np.mean(np.abs(first_score-second_score))
        )
        scored_sentences.append(scored_sentence)

    # Sort sentences descending by average different score assigned to individual time steps by the given metric
    sorted_scored_sentences = sorted(scored_sentences, key=lambda sen: sen.diff, reverse=True)

    # Plot most different samples
    for i, scored_sentence in enumerate(sorted_scored_sentences[:int(samples/2)]):
        plot_stepwise_metric_line(scored_sentence, model_names, img_dir, opt, i, pdf=pdf)

    # Sample some more random samples for comparison
    for i, scored_sentence in enumerate(random.sample(sorted_scored_sentences[int(samples/2):], k=int(samples/2))):
        plot_stepwise_metric_line(scored_sentence, model_names, img_dir, opt, i + int(samples/2), pdf=pdf)

    if pdf is not None:
        pdf.close()


def plot_stepwise_metric_line(scored_sentence,  model_names, img_dir, opt, num, pdf=None):
    tokens = scored_sentence.src

    def pad_data(data):
        return np.concatenate(([[None]], data, [[None]]), axis=1).squeeze(0)

    # TODO: Make this metric agnostic
    first_scores = pad_data(scored_sentence.first_scores)
    second_scores = pad_data(scored_sentence.second_scores)
    x = range(len(tokens) + 1)

    fig, ax = plt.subplots()

    # Plot data
    first_high = (scored_sentence.first_scores + scored_sentence.first_std).squeeze(0)
    first_low = (scored_sentence.first_scores - scored_sentence.first_std).squeeze(0)
    second_high = (scored_sentence.second_scores + scored_sentence.second_std).squeeze(0)
    second_low = (scored_sentence.second_scores - scored_sentence.second_std).squeeze(0)
    ax.plot(x, first_scores, label="Baseline", color="tab:blue")
    plt.fill_between(x[1:-1], first_high,  scored_sentence.first_scores.squeeze(0), alpha=0.4, color="tab:blue")
    plt.fill_between(x[1:-1],  scored_sentence.first_scores.squeeze(0), first_low, alpha=0.4, color="tab:blue")

    ax.plot(x, second_scores, label="Attention", color="tab:orange")
    plt.fill_between(x[1:-1], second_high, scored_sentence.second_scores.squeeze(0), alpha=0.4, color="tab:orange")
    plt.fill_between(x[1:-1], scored_sentence.second_scores.squeeze(0), second_low, alpha=0.4, color="tab:orange")

    ax.plot(x, [1] * (len(tokens) + 1), linestyle="dashed", color="gray")
    ax.set_ylim(top=1.8, bottom=0.2)
    plt.xticks(x, tokens + [""], fontsize=13)

    # Arrows and text
    plt.text(0.94, 0.35, "Integrate", transform=plt.gca().transAxes, rotation=90, fontsize=11)
    plt.text(0.94, 0.8, "Maintain", transform=plt.gca().transAxes, rotation=90, fontsize=11)
    ax.arrow(
        len(tokens) - 0.2, 1.025, 0, 0.7, alpha=0.6, head_width=0.1, length_includes_head=True, color="black",
        overhang=0.75, head_length=0.05
    )
    ax.arrow(
        len(tokens) - 0.2, 0.975, 0, -0.7, alpha=0.6, head_width=0.1, length_includes_head=True, color="black",
        overhang=0.75, head_length=0.05
    )

    # Draw vertical lines
    for x_ in range(len(tokens)):
        plt.axvline(x=x_, alpha=0.8, color="gray", linewidth=0.25)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    plt.legend(loc="upper left")
    plt.tight_layout()

    if pdf is None:
        plt.savefig(f"{img_dir}/{opt.metrics[0]}_{num}.png")
    else:
        pdf.savefig(fig)

    plt.close()


def plot_stepwise_metric_heat(scored_sentence, model_names, img_dir, opt, num):
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
        ax.text(i, 0, round(first_scores[i], 3), ha="center", va="center", color="black" if 1.05 > first_scores[i] > 0.95 else "w")
        ax.text(i, 1, round(second_scores[i], 3), ha="center", va="center", color="black" if 1.05 > second_scores[i] > 0.95 else "w")

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    cbar = ax.figure.colorbar(img, ax=ax, shrink=0.315)
    cbar.ax.set_ylabel("Integration Ratio", rotation=-90, va="bottom")

    fig.tight_layout()
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
    parser.add_argument("--models", nargs="+", help="List of paths to models used to conduct analyses.")
    parser.add_argument("--break_after", type=int)

    return parser


if __name__ == "__main__":
    qualitative_analysis(
        model_names=("Baseline", "Attention"), samples=20, img_dir="./img/qualitative",
        pdf_path="./img/qualitative/qualitative.pdf"
    )