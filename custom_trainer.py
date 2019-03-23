"""
Subclass of the supervised trainer with a pre-training functionality.
"""

from collections import defaultdict
import os
import shutil

import torchtext
import torch

import machine
from machine.trainer import SupervisedTrainer
from machine.loss import NLLLoss
from machine.util.log import Log
from machine.util.checkpoint import Checkpoint

from incremental_loss import AnticipationEmbeddingLoss, AnticipationLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CONSTANTS
ANTICIPATION_LOSSES = (AnticipationEmbeddingLoss, AnticipationLoss)

# TODO: Set non-anticipation losses to 0 when pre-training in combination with not calling backward


class SupervisedPreTrainer(SupervisedTrainer):

    def __init__(self, anticipation_pretraining, expt_dir='experiment', loss=[NLLLoss()], loss_weights=None, metrics=[],
                 batch_size=64, eval_batch_size=128, random_seed=None, checkpoint_every=100, print_every=100):
        # Number of epochs given to pretrain with only anticipation loss
        self.anticipation_pretraining = anticipation_pretraining
        self.current_epoch = None

        super().__init__(expt_dir, loss, loss_weights, metrics, batch_size, eval_batch_size, random_seed,
                         checkpoint_every, print_every)

    def _train_batch(self, input_variable, input_lengths,
                     target_variable, model, teacher_forcing_ratio):

        # Forward propagation
        decoder_outputs, decoder_hidden, other = model(
            input_variable, input_lengths, target_variable, teacher_forcing_ratio=teacher_forcing_ratio)

        losses = self.evaluator.compute_batch_loss(
            decoder_outputs, decoder_hidden, other, target_variable)

        # Backward propagation
        for i, loss in enumerate(losses, 0):

            # Skip calling backward on a non-anticipation loss if pre-training
            if self.ignore_loss(loss):
                continue

            loss.scale_loss(self.loss_weights[i])
            loss.backward(retain_graph=True)

        self.optimizer.step()
        model.zero_grad()

        return losses


    @property
    def pretraining(self):
        return self.current_epoch < self.anticipation_pretraining

    def ignore_loss(self, loss):
        return type(loss) not in ANTICIPATION_LOSSES and self.pretraining

    def _train_epoches(self, data, model, n_epochs, start_epoch, start_step,
                       dev_data=None, monitor_data=[], teacher_forcing_ratio=0,
                       top_k=5):
        log = self.logger

        pretraining_end_logged = False
        if self.anticipation_pretraining > 0:
            log.info(f"Pre-training with anticipation loss for {self.anticipation_pretraining} epochs.")

        print_loss_total = defaultdict(float)  # Reset every print_every
        epoch_loss_total = defaultdict(float)  # Reset every epoch
        epoch_loss_avg = defaultdict(float)
        print_loss_avg = defaultdict(float)

        batch_iterator = torchtext.data.BucketIterator(
            dataset=data, batch_size=self.batch_size,
            sort=False, sort_within_batch=True,
            sort_key=lambda x: len(x.src),
            device=device, repeat=False)

        steps_per_epoch = len(batch_iterator)
        total_steps = steps_per_epoch * n_epochs

        step = start_step
        step_elapsed = 0

        # store initial model to be sure at least one model is stored
        val_data = dev_data or data
        losses, metrics = self.evaluator.evaluate(
            model, val_data, self.get_batch_data)

        total_loss, log_msg, model_name = self.get_losses(
            losses, metrics, step)
        log.info(log_msg)

        logs = Log()
        loss_best = top_k * [total_loss]
        best_checkpoints = top_k * [None]
        best_checkpoints[0] = model_name

        Checkpoint(model=model,
                   optimizer=self.optimizer,
                   epoch=start_epoch, step=start_step,
                   input_vocab=data.fields[machine.src_field_name].vocab,
                   output_vocab=data.fields[machine.tgt_field_name].vocab).save(self.expt_dir, name=model_name)

        for epoch in range(start_epoch, n_epochs + 1):
            log.info("Epoch: %d, Step: %d" % (epoch, step))
            self.current_epoch = epoch

            if epoch > self.anticipation_pretraining and not pretraining_end_logged:
                log.info("Pretraining with anticipation loss has ended.")
                pretraining_end_logged = True

            batch_generator = batch_iterator.__iter__()

            # consuming seen batches from previous training
            for _ in range((epoch - 1) * steps_per_epoch, step):
                next(batch_generator)

            model.train(True)
            for batch in batch_generator:
                step += 1
                step_elapsed += 1

                input_variables, input_lengths, target_variables = self.get_batch_data(
                    batch)

                losses = self._train_batch(input_variables, input_lengths.tolist(
                ), target_variables, model, teacher_forcing_ratio)

                # Record average loss
                for loss in losses:
                    if self.ignore_loss(loss):
                        continue

                    name = loss.log_name
                    print_loss_total[name] += loss.get_loss()
                    epoch_loss_total[name] += loss.get_loss()

                # print log info according to print_every parm
                if step % self.print_every == 0 and step_elapsed > self.print_every:
                    for loss in losses:
                        if self.ignore_loss(loss):
                            continue

                        name = loss.log_name
                        print_loss_avg[name] = print_loss_total[name] / \
                            self.print_every
                        print_loss_total[name] = 0

                    m_logs = {}
                    train_losses, train_metrics = self.evaluator.evaluate(
                        model, data, self.get_batch_data)
                    train_loss, train_log_msg, model_name = self.get_losses(
                        train_losses, train_metrics, step)
                    logs.write_to_log('Train', train_losses,
                                      train_metrics, step)
                    logs.update_step(step)

                    m_logs['Train'] = train_log_msg

                    # compute vals for all monitored sets
                    for m_data in monitor_data:
                        losses, metrics = self.evaluator.evaluate(
                            model, monitor_data[m_data], self.get_batch_data)
                        total_loss, log_msg, model_name = self.get_losses(
                            losses, metrics, step)
                        m_logs[m_data] = log_msg

                        logs.write_to_log(m_data, losses, metrics, step)

                    all_losses = ' '.join(
                        ['%s:\t %s\n' % (os.path.basename(name), m_logs[name]) for name in m_logs])

                    log_msg = 'Progress %d%%, %s' % (
                        step / total_steps * 100,
                        all_losses)

                    log.info(log_msg)

                # check if new model should be saved
                if step % self.checkpoint_every == 0 or step == total_steps:
                    # compute dev loss
                    losses, metrics = self.evaluator.evaluate(
                        model, val_data, self.get_batch_data)
                    total_loss, log_msg, model_name = self.get_losses(
                        losses, metrics, step)

                    max_eval_loss = max(loss_best)
                    if total_loss < max_eval_loss:
                        index_max = loss_best.index(max_eval_loss)
                        # rm prev model
                        if best_checkpoints[index_max] is not None:
                            shutil.rmtree(os.path.join(
                                self.expt_dir, best_checkpoints[index_max]))
                        best_checkpoints[index_max] = model_name
                        loss_best[index_max] = total_loss

                        # save model
                        Checkpoint(model=model,
                                   optimizer=self.optimizer,
                                   epoch=epoch, step=step,
                                   input_vocab=data.fields[machine.src_field_name].vocab,
                                   output_vocab=data.fields[machine.tgt_field_name].vocab).save(self.expt_dir, name=model_name)

            if step_elapsed == 0:
                continue

            for loss in losses:
                epoch_loss_avg[loss.log_name] = epoch_loss_total[loss.log_name] / \
                    min(steps_per_epoch, step - start_step)
                epoch_loss_total[loss.log_name] = 0

            if dev_data is not None:
                losses, metrics = self.evaluator.evaluate(
                    model, dev_data, self.get_batch_data)
                loss_total, log_, model_name = self.get_losses(
                    losses, metrics, step)

                # TODO check if this makes sense!
                self.optimizer.update(loss_total, epoch)
                log_msg += ", Dev set: " + log_
                model.train(mode=True)
            else:
                # TODO check if this makes sense!
                self.optimizer.update(epoch_loss_avg, epoch)

            log.info(log_msg)

        return logs

