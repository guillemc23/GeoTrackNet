# Copyright 2017 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

from __future__ import absolute_import, division, print_function

import logging
import os
import time

import tensorflow as tf

tf.compat.v1.disable_v2_behavior()

import migration.bounds as bounds
from migration.get_config_old import config
from migration.models import vrnn
from migration.models.dataset import datasets as datasets


def create_dataset_and_model(config, shuffle, repeat):
    """

    """
    inputs, targets, mmsis, time_starts, time_ends, lengths, mean = datasets.create_AIS_dataset(config.trainingset_path,
                                                          os.path.join(os.path.dirname(config.trainingset_path),"training_mean.pkl"),
                                                          config.batch_size,
                                                          config.onehot_lat_bins,
                                                          config.onehot_lon_bins,
                                                          config.onehot_sog_bins,
                                                          config.onehot_cog_bins,
                                                          shuffle=shuffle,
                                                          repeat_indefinitely=repeat)
    # Convert the mean of the training set to logit space so it can be used to
    # initialize the bias of the generative distribution.
    generative_bias_init = -tf.math.log(1. / tf.clip_by_value(mean, 0.0001, 0.9999) - 1)
    generative_distribution_class = vrnn.ConditionalBernoulliDistribution
    model = vrnn.create_vrnn(inputs.get_shape().as_list()[2],
                             config.latent_size,
                             generative_distribution_class,
                             generative_bias_init=generative_bias_init,
                             raw_sigma_bias=0.5)
    return inputs, targets, mmsis, time_starts, time_ends, lengths, model


def restore_checkpoint_if_exists(saver, sess, logdir):
    """Looks for a checkpoint and restores the session from it if found.
    Args:
      saver: A tf.train.Saver for restoring the session.
      sess: A TensorFlow session.
      logdir: The directory to look for checkpoints in.
    Returns:
      True if a checkpoint was found and restored, False otherwise.
    """
    checkpoint = tf.train.get_checkpoint_state(logdir)
    if checkpoint:
        checkpoint_name = os.path.basename(checkpoint.model_checkpoint_path)
        full_checkpoint_path = os.path.join(logdir, checkpoint_name)
        saver.restore(sess, full_checkpoint_path)
        return True
    return False


def wait_for_checkpoint(saver, sess, logdir):
    while True:
        if restore_checkpoint_if_exists(saver, sess, logdir):
            break
        else:
            tf.compat.v1.logging.info("Checkpoint not found in %s, sleeping for 60 seconds."
                      % logdir)
            time.sleep(60)


def run_train(config):

    def create_logging_hook(step, bound_value):
        """Creates a logging hook that prints the bound value periodically."""
        bound_label = config.bound + " bound"
        if config.normalize_by_seq_len:
            bound_label += " per timestep"
        else:
            bound_label += " per sequence"
        def summary_formatter(log_dict):
            return "Step %d, %s: %f" % (
                        log_dict["step"], bound_label, log_dict["bound_value"])
        logging_hook = tf.estimator.LoggingTensorHook(
                                {"step": step,
                                 "bound_value": bound_value},
                                every_n_iter=config.summarize_every,
                                formatter=summary_formatter)
        return logging_hook

    def create_loss():
        """Creates the loss to be optimized.
        Returns:
            bound: A float Tensor containing the value of the bound that is
                   being optimized.
            loss: A float Tensor that when differentiated yields the gradients
                to apply to the model. Should be optimized via gradient descent.
        """
        inputs, targets, mmsis, time_starts, time_ends, lengths, model = create_dataset_and_model(config,
                                                               shuffle=True,
                                                               repeat=True)
        # Compute lower bounds on the log likelihood.
        if config.bound == "elbo":
            ll_per_seq, _, _, _ = bounds.elbo(model,
                                              (inputs, targets),
                                              lengths,
                                              num_samples=1)
        elif config.bound == "fivo":
            ll_per_seq, _, _, _, _ = bounds.fivo(model,
                                                 (inputs, targets),
                                                 lengths,
                                                 num_samples=config.num_samples,
                                                 resampling_criterion=bounds.ess_criterion)
        # Compute loss scaled by number of timesteps.
        ll_per_t = tf.reduce_mean(input_tensor=ll_per_seq / tf.cast(lengths, dtype=tf.float32))
        ll_per_seq = tf.reduce_mean(input_tensor=ll_per_seq)

        tf.compat.v1.summary.scalar("train_ll_per_seq", ll_per_seq)
        tf.compat.v1.summary.scalar("train_ll_per_t", ll_per_t)

        if config.normalize_by_seq_len:
            return ll_per_t, -ll_per_t
        else:
            return ll_per_seq, -ll_per_seq

    def create_graph():
        """Creates the training graph."""
        global_step = tf.compat.v1.train.get_or_create_global_step()
        bound, loss = create_loss()
        opt = tf.compat.v1.train.AdamOptimizer(config.learning_rate)
        grads = opt.compute_gradients(loss, var_list=tf.compat.v1.trainable_variables())
        train_op = opt.apply_gradients(grads, global_step=global_step)
        return bound, train_op, global_step

    device = tf.compat.v1.train.replica_device_setter(ps_tasks=config.ps_tasks)
    with tf.Graph().as_default():
        if config.random_seed: tf.compat.v1.set_random_seed(config.random_seed)
        with tf.device(device):
            bound, train_op, global_step = create_graph()
            log_hook = create_logging_hook(global_step, bound)
            start_training = not config.stagger_workers
            with tf.compat.v1.train.MonitoredTrainingSession(master=config.master,
                                                   is_chief=config.task == 0,
                                                   hooks=[log_hook],
                                                   checkpoint_dir=config.logdir,
                                                   save_checkpoint_secs=120,
                                                   save_summaries_steps=config.summarize_every,
                                                   log_step_count_steps=config.summarize_every) as sess:
                cur_step = -1
                while True:
                    if sess.should_stop() or cur_step > config.max_steps: break
                    if config.task > 0 and not start_training:
                        cur_step = sess.run(global_step)
                        tf.compat.v1.logging.info("task %d not active yet, sleeping at step %d" %
                            (config.task, cur_step))
                        time.sleep(30)
                        if cur_step >= config.task * 1000:
                            start_training = True
                    else:
                        _, cur_step = sess.run([train_op, global_step])
#                         _, cur_step = sess.run([train_op, global_step])

if __name__ == '__main__':
    print(config.trainingset_path)
    fh = logging.FileHandler(os.path.join(config.logdir,config.log_filename+".log"))
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
    # get TF logger
    logger = logging.getLogger('tensorflow')
    logger.addHandler(fh)
    run_train(config)