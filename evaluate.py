import os
import argparse
import logging

import torch
import torchtext

from machine.loss.loss import NLLLoss
from machine.dataset import SourceField, TargetField
from machine.evaluator import Evaluator
from machine.util.checkpoint import Checkpoint
from machine.trainer import SupervisedTrainer
from machine.metrics import WordAccuracy, SequenceAccuracy

try:
    raw_input          # Python 2
except NameError:
    raw_input = input  # Python 3

parser = argparse.ArgumentParser()

parser.add_argument('--checkpoint_path', help='Give the checkpoint path from which to load the model')
parser.add_argument('--test_data', help='Path to test data')
parser.add_argument('--cuda_device', default=0, type=int, help='set cuda device to use')
parser.add_argument('--max_len', type=int, help='Maximum sequence length', default=50)
parser.add_argument('--batch_size', type=int, help='Batch size', default=32)

opt = parser.parse_args()

if torch.cuda.is_available():
        print("Cuda device set to %i" % opt.cuda_device)
        torch.cuda.set_device(opt.cuda_device)

#################################################################################
# load model

logging.info("loading checkpoint from {}".format(os.path.join(opt.checkpoint_path)))
checkpoint = Checkpoint.load(opt.checkpoint_path)
seq2seq = checkpoint.model
input_vocab = checkpoint.input_vocab
output_vocab = checkpoint.output_vocab

############################################################################
# Prepare dataset and loss
src = SourceField()
tgt = TargetField()
src.vocab = input_vocab
tgt.vocab = output_vocab
max_len = opt.max_len


def len_filter(example):
    return len(example.src) <= max_len and len(example.tgt) <= max_len

# generate test set
test = torchtext.data.TabularDataset(
    path=opt.test_data, format='tsv',
    fields=[('src', src), ('tgt', tgt)],
    filter_pred=len_filter
)

# Prepare loss
weight = torch.ones(len(output_vocab))
pad = output_vocab.stoi[tgt.pad_token]
loss = NLLLoss(pad)
metrics = [WordAccuracy(pad), SequenceAccuracy(pad)]
if torch.cuda.is_available():
    loss.cuda()

#################################################################################
# Evaluate model on test set

evaluator = Evaluator(loss=[loss], metrics=metrics, batch_size=opt.batch_size)
losses, metrics = evaluator.evaluate(seq2seq, test, SupervisedTrainer.get_batch_data)

print(["{}: {:6f}".format(type(metric).__name__, metric.get_val()) for metric in metrics])
