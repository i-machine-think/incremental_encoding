"""
Define custom training script for incremental models used in this project.
"""


import os

import torch
from machine.models import EncoderRNN, DecoderRNN, Seq2seq
from train_model import init_argparser, validate_options, init_logging, prepare_dataset, initialize_model, \
    prepare_losses_and_metrics, create_trainer, load_model_from_checkpoint

from incremental import AnticipatingEncoderRNN, IncrementalSeq2Seq, AnticipatingEmbeddingEncoderRNN
from incremental_loss import AnticipationLoss, AnticipationEmbeddingLoss

# GLOBALS
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CONSTANTS
IGNORE_INDEX = -1


def train_incremental_model():
    parser = init_argparser()
    parser = add_incremental_parser_args(parser)  # Add extra parser arguments used in this project
    opt = parser.parse_args()
    opt = validate_options(parser, opt)

    # Prepare logging and data set
    init_logging(opt)
    src, tgt, train, dev, monitor_data = prepare_dataset(opt)

    # Prepare model
    if opt.load_checkpoint is not None:
        seq2seq, input_vocab, output_vocab = load_model_from_checkpoint(
            opt, src, tgt)
    else:
        seq2seq, input_vocab, output_vocab = initialize_incremental_model(
            opt, src, tgt, train)

    pad = output_vocab.stoi[tgt.pad_token]
    eos = tgt.eos_id
    sos = tgt.SYM_EOS
    unk = tgt.unk_token

    # Prepare training
    losses, loss_weights, metrics = prepare_losses_and_metrics(
        opt, pad, unk, sos, eos, input_vocab, output_vocab)
    losses, loss_weights = add_incremental_losses(opt, losses, loss_weights)
    checkpoint_path = os.path.join(
        opt.output_dir, opt.load_checkpoint) if opt.resume else None
    trainer = create_trainer(opt, losses, loss_weights, metrics)

    # Train
    seq2seq, logs = trainer.train(seq2seq, train,
                                  num_epochs=opt.epochs, dev_data=dev, monitor_data=monitor_data, optimizer=opt.optim,
                                  teacher_forcing_ratio=opt.teacher_forcing_ratio, learning_rate=opt.lr,
                                  resume=opt.resume, checkpoint_path=checkpoint_path)

    if opt.write_logs:
        output_path = os.path.join(opt.output_dir, opt.write_logs)
        logs.write_to_file(output_path)


def add_incremental_parser_args(parser):
    parser.add_argument('--scale_anticipation_loss', type=float, default=1.0,
                        help="Scale the anticipation loss with some factor,")
    parser.add_argument('--anticipation_loss', choices=["normal", "embeddings"],
                        help='Indicate whether an additional loss term should be imposed on the encoder.')
    return parser


def initialize_incremental_model(opt, src, tgt, train):
    src.build_vocab(train, max_size=opt.src_vocab)
    tgt.build_vocab(train, max_size=opt.tgt_vocab)
    input_vocab = src.vocab
    output_vocab = tgt.vocab

    # Initialize model
    hidden_size = opt.hidden_size
    decoder_hidden_size = hidden_size * 2 if opt.bidirectional else hidden_size

    # Pick special classes for encoder / decoder if anticipation loss is used
    # -> they require different information to calculate the loss
    encoder_classes = {
        None: EncoderRNN,
        "normal": AnticipatingEncoderRNN,
        "embeddings": AnticipatingEmbeddingEncoderRNN
    }
    # Set up for future changes
    decoder_classes = {
        None: DecoderRNN,
        "normal": DecoderRNN,
        "embeddings": DecoderRNN
    }

    encoder_cls = encoder_classes[opt.anticipation_loss]
    decoder_cls = decoder_classes[opt.anticipation_loss]

    encoder = encoder_cls(len(src.vocab), opt.max_len, hidden_size,
                         opt.embedding_size,
                         dropout_p=opt.dropout_p_encoder,
                         n_layers=opt.n_layers,
                         bidirectional=opt.bidirectional,
                         rnn_cell=opt.rnn_cell,
                         variable_lengths=True)
    decoder = decoder_cls(len(tgt.vocab), opt.max_len, decoder_hidden_size,
                         dropout_p=opt.dropout_p_decoder,
                         n_layers=opt.n_layers,
                         use_attention=opt.attention,
                         attention_method=opt.attention_method,
                         bidirectional=opt.bidirectional,
                         rnn_cell=opt.rnn_cell,
                         eos_id=tgt.eos_id, sos_id=tgt.sos_id)

    if opt.anticipation_loss is None:
        seq2seq = Seq2seq(encoder, decoder)
    else:
        seq2seq = IncrementalSeq2Seq(encoder, decoder, use_embeddings=(opt.anticipation_loss == "embeddings"))

    seq2seq.to(device)

    return seq2seq, input_vocab, output_vocab


def add_incremental_losses(opt, losses, loss_weights):
    incremental_losses = []

    # Use cross-entropy loss to compare the output distribution against the actual next word in the sequence
    if opt.anticipation_loss == "normal":
        anticipation_loss = AnticipationLoss(ignore_index=IGNORE_INDEX)
        incremental_losses.append(anticipation_loss)
        loss_weights.append(opt.scale_anticipation_loss)

    # Use euclidian distance to measure the difference between the embedding of the most likely predicted word
    # to the embedding of the actual next word in the sequence
    elif opt.anticipation_loss == "embeddings":
        anticipation_loss = AnticipationEmbeddingLoss()
        incremental_losses.append(anticipation_loss)
        loss_weights.append(opt.scale_anticipation_loss)

    for loss in incremental_losses:
        loss.to(device)

    losses.extend(incremental_losses)

    return losses, loss_weights


if __name__ == "__main__":
    train_incremental_model()
