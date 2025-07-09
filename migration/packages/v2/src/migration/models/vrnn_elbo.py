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
# ==============================================================================

"""VRNN classes."""


import tensorflow as tf
from tensorflow.keras import Model

import migration.nested_utils as nested
from migration.models.distributions import (
  ConditionalBernoulliDistribution,
  ConditionalNormalDistribution,
  NormalApproximatePosterior,
)
from migration.models.mlp import build_mlp
from migration.models.vrnn_cell import VRNNCell

_DEFAULT_INITIALIZERS = {"w": tf.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"),
                         "b": tf.zeros_initializer()}


class VRNN(Model):

  def __init__(self,
               data_size : int,
    latent_size : int,
    generative_class,
    sigma_min=0.0,
    raw_sigma_bias=0.25,
    generative_bias_init=0.0,
    initializers=None,
    random_seed=None, 
    num_samples=1,
    parallel_iterations=30,
    swap_memory=True, 
    name='vrnn'):

    
    """Creates a VRNN cell.

    Args:
      rnn_cell: A subclass of tf.nn.rnn_cell.RNNCell that will form the
        deterministic backbone of the VRNN. The inputs to the RNN will be the
        encoded latent state of the previous timestep with shape
        [batch_size, encoded_latent_size] as well as the encoded input of the
        current timestep, a Tensor of shape [batch_size, encoded_data_size].
      data_feat_extractor: A callable that accepts a batch of data x_t and
        'encodes' it, e.g. runs it through a fully connected network. Must
        accept as argument the inputs x_t, a Tensor of the shape
        [batch_size, data_size] and return a Tensor of shape
        [batch_size, encoded_data_size]. This callable will be called multiple
        times in the VRNN cell so if scoping is not handled correctly then
        multiple copies of the variables in this network could be made. It is
        recommended to use a snt.nets.MLP module, which takes care of this for
        you.
      latent_feat_extractor: A callable that accepts a latent state z_t and
        'encodes' it, e.g. runs it through a fully connected network. Must
        accept as argument a Tensor of shape [batch_size, latent_size] and
        return a Tensor of shape [batch_size, encoded_latent_size].
        This callable must also have the property 'output_size' defined,
        returning encoded_latent_size.
      prior: A callable that implements the prior p(z_t|h_t). Must accept as
        argument the previous RNN hidden state and return a
        tfp.distributions.Normal distribution conditioned on the input.
      approx_posterior: A callable that implements the approximate posterior
        q(z_t|h_t,x_t). Must accept as arguments the encoded target of the
        current timestep and the previous RNN hidden state. Must return
        a tfp.distributions.Normal distribution conditioned on the
        inputs.
      generative: A callable that implements the generative distribution
        p(x_t|z_t, h_t). Must accept as arguments the encoded latent state
        and the RNN hidden state and return a subclass of
        tfp.distributions.Distribution that can be used to evaluate
        the logprob of the targets.
      random_seed: The seed for the random ops. Used mainly for testing.
      name: The name of this VRNN.
    """
    super(VRNN, self).__init__(name=name)
    
    self.num_samples = num_samples
    self.parallel_iterations = parallel_iterations
    self.swap_memory = swap_memory

    if initializers is None:
        # TODO: review initializers
        initializers = _DEFAULT_INITIALIZERS


    data_feat_extractor = build_mlp(
        layer_sizes=[latent_size, latent_size],
        initializers=_DEFAULT_INITIALIZERS,   
        name='data_feat_extractor'

    )
    latent_feat_extractor = build_mlp(
        layer_sizes=[latent_size, latent_size],
        initializers=_DEFAULT_INITIALIZERS,
        name='latent_feat_extractor'

    )
    prior = ConditionalNormalDistribution(
        size=latent_size,
        hidden_layer_size=latent_size,
        sigma_min=sigma_min,
        raw_sigma_bias=raw_sigma_bias,
        initializers=initializers)
    approx_posterior = NormalApproximatePosterior(
        size=latent_size,
        hidden_layer_size=latent_size,
        sigma_min=sigma_min,
        raw_sigma_bias=raw_sigma_bias,
        initializers=initializers)
    if generative_class == ConditionalBernoulliDistribution:
        generative = ConditionalBernoulliDistribution(
            size=data_size,
            hidden_layer_size=latent_size,
            initializers=initializers,
            bias_init=generative_bias_init)
    else:
        generative = ConditionalNormalDistribution(
            size=data_size,
            hidden_layer_size=latent_size,
            initializers=initializers)
        
    rnn_cell = tf.keras.layers.LSTMCell(latent_size, kernel_initializer=initializers['w'])
    self.vrnn_cell =  VRNNCell(rnn_cell, data_feat_extractor, latent_feat_extractor, latent_size,
                  prior, approx_posterior, generative, random_seed=random_seed)
  
  def call(self, inputs, seq_lengths):
    """Computes one timestep of the VRNN.

    Args:
      observations: The observations at the current timestep, a tuple
        containing the model inputs and targets as Tensors of shape
        [batch_size, data_size].
      state: The current state of the VRNN
      mask: Tensor of shape [batch_size], 1.0 if the current timestep is active
        active, 0.0 if it is not active.
      return_value: "logits" or "probs"

    Returns:
      log_q_z: The logprob of the latent state according to the approximate
        posterior.
      log_p_z: The logprob of the latent state according to the prior.
      log_p_x_given_z: The conditional log-likelihood, i.e. logprob of the
        observation according to the generative distribution.
      kl: The analytic kl divergence from q(z) to p(z).
      state: The new state of the VRNN.
    """
    batch_size = tf.shape(input=seq_lengths)[0]
    max_seq_len = tf.reduce_max(input_tensor=seq_lengths)
    seq_mask = tf.transpose(
            a=tf.sequence_mask(seq_lengths, maxlen=max_seq_len, dtype=tf.float32),
            perm=[1, 0])
    # not accessed
    # if num_samples > 1:
    #     inputs, seq_mask = nested.tile_tensors([inputs, seq_mask], [1, num_samples])
    # not accessed

    # tensorarray of len t, elem (B, A)
    inputs_ta, mask_ta = nested.tas_for_tensors([inputs, seq_mask], max_seq_len)

    t0 = tf.constant(0, tf.int32)
    init_states = self.vrnn_cell.zero_state(batch_size * self.num_samples, tf.float32)
    # TODO: commented
    # init_inputs, init_mask = nested.read_tas([inputs_ta, mask_ta], t0)
    
    log_weights_acc = tf.zeros([self.num_samples, batch_size], dtype=tf.float32)
    kl_acc = tf.zeros([self.num_samples * batch_size], dtype=tf.float32)
    accs = (log_weights_acc, kl_acc)

    def while_predicate(t, *unused_args):
        return t < max_seq_len

    def while_step(t, rnn_state, accs):
        """Implements one timestep of IWAE computation."""
        log_weights_acc, kl_acc = accs
        cur_inputs, cur_mask = nested.read_tas([inputs_ta, mask_ta], t)
        # Run the cell for one step.
        log_q_z, log_p_z, log_p_x_given_z, kl, new_state, new_rnn_out\
                                                     = self.vrnn_cell(cur_inputs,
                                                            rnn_state,
                                                            cur_mask,
                                                            )
        # Compute the incremental weight and use it to update the current
        # accumulated weight.
        kl_acc += kl * cur_mask
        log_alpha = (log_p_x_given_z + log_p_z - log_q_z) * cur_mask
        log_alpha = tf.reshape(log_alpha, [self.num_samples, batch_size])
        log_weights_acc += log_alpha 
        # Update the  Tensorarrays and accumulators.
        new_accs = (log_weights_acc, kl_acc)
        return t + 1, new_state, new_accs
    
    # AutoGraph converts while-loop to tf.while_loop().
    _, _, accs = tf.while_loop(cond=while_predicate,
                                    body=while_step,
                                    loop_vars=(t0, init_states, accs),
                                    parallel_iterations=self.parallel_iterations,
                                    swap_memory=self.swap_memory)
    
    ## Here log_weights is acc log_weights
    final_log_weights, _ = accs
    log_p_hat = (tf.reduce_logsumexp(input_tensor=final_log_weights, axis=0) -
                                 tf.math.log(tf.cast(self.num_samples, dtype=tf.float32)))
    return log_p_hat


