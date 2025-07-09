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

import migration.bounds.model_with_elbo as RNNlayer
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

    if config.random_seed: tf.random.set_seed(config.random_seed)

    inputs, targets, _, _, _, lengths, model = create_dataset_and_model(config,
                                                               shuffle=True,
                                                               repeat=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=config.learning_rate)

    @tf.function
    def train_step(x,y):
        with tf.GradientTape() as tape:
            bound, _, _,_ = RNNlayer(model, (x, y),
                                    lengths,
                                    num_samples=1)
            

            # Compute lower bounds on the log likelihood.
            bound = tf.reduce_mean(input_tensor=bound / tf.cast(lengths, dtype=tf.float32))
            loss = -bound
        grads = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        return loss
    for epoch in range(config.max_steps):
        loss_value = train_step(inputs, targets)


if __name__ == '__main__':
    print(config.trainingset_path)
    fh = logging.FileHandler(os.path.join(config.logdir,config.log_filename+".log"))
    # get TF logger
    logger = logging.getLogger('tensorflow')
    logger.addHandler(fh)
    run_train(config)