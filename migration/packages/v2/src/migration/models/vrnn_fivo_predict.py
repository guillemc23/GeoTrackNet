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
import tensorflow_probability as tfp
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

def ess_criterion(num_samples, log_ess, unused_t):
    """A criterion that resamples based on effective sample size."""
    return log_ess <= tf.math.log(num_samples / 2.0)

class VRNNboundFIVO(Model):

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
    super(VRNNboundFIVO, self).__init__(name=name)
    
    self.num_samples = num_samples
    self.parallel_iterations = parallel_iterations
    self.swap_memory = swap_memory
    self.resampling_criterion = ess_criterion
    self.random_seed = random_seed

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
    batch_size = tf.shape(input=seq_lengths)[0]
    max_seq_len = tf.reduce_max(input_tensor=seq_lengths)
    
    seq_mask = tf.transpose(
            a=tf.sequence_mask(seq_lengths, maxlen=max_seq_len, dtype=tf.float32),
            perm=[1, 0])
    # Each sequence in the batch will be the input data for a different
    # particle filter. The batch will be laid out as:
    #   particle 1 of particle filter 1
    #   particle 1 of particle filter 2
    #   ...
    #   particle 1 of particle filter batch_size
    #   particle 2 of particle filter 1
    #   ...
    #   particle num_samples of particle filter batch_size
    if self.num_samples > 1:
        inputs, seq_mask = nested.tile_tensors([inputs, seq_mask], [1, self.num_samples])
    # inputs: [max_seq_len, batch_size*num_samples, ...] (Duong)
    inputs_ta, mask_ta = nested.tas_for_tensors([inputs, seq_mask], max_seq_len)

    t0 = tf.constant(0, tf.int32)
    init_states = self.vrnn_cell.zero_state(batch_size * self.num_samples, tf.float32)
    log_weights_ta = tf.TensorArray(tf.float32, max_seq_len, name='log_weights_ta')
    log_weights_acc = tf.zeros([self.num_samples, batch_size], dtype=tf.float32)

    def while_predicate(t, *unused_args):
        return t < max_seq_len

    def while_step(t, rnn_state, log_weights_ta, log_weights_acc):
        """Implements one timestep of FIVO computation."""
        cur_inputs, cur_mask = nested.read_tas([inputs_ta, mask_ta], t)
        # cur_inputs: slice at time t, [batch_size*num_samples, ...]  (Duong)
        # Run the cell for one step.
        log_q_z, log_p_z, log_p_x_given_z, _, new_state, _\
                                                         = self.vrnn_cell(cur_inputs,
                                                                rnn_state,
                                                                cur_mask,
                                                                )
        # Compute the incremental weight and use it to update the current
        # accumulated weight.
        log_alpha = (log_p_x_given_z + log_p_z - log_q_z) * cur_mask # ELBO (Duong)
        log_alpha = tf.reshape(log_alpha, [self.num_samples, batch_size])
        log_weights_acc += log_alpha
        # Calculate the effective sample size.
        ess_num = 2 * tf.reduce_logsumexp(input_tensor=log_weights_acc, axis=0)
        ess_denom = tf.reduce_logsumexp(input_tensor=2 * log_weights_acc, axis=0)
        log_ess = ess_num - ess_denom
        # Calculate the ancestor indices via resampling. Because we maintain the
        # log unnormalized weights, we pass the weights in as logits, allowing
        # the distribution object to apply a softmax and normalize them.
        resampling_dist = tfp.distributions.Categorical(
                            logits=tf.transpose(a=log_weights_acc, perm=[1, 0]))
        ancestor_inds = tf.stop_gradient(
                resampling_dist.sample(sample_shape=self.num_samples, seed=self.random_seed))
        # Because the batch is flattened and laid out as discussed
        # above, we must modify ancestor_inds to index the proper samples.
        # The particles in the ith filter are distributed every batch_size rows
        # in the batch, and offset i rows from the top. So, to correct the indices
        # we multiply by the batch_size and add the proper offset. Crucially,
        # when ancestor_inds is flattened the layout of the batch is maintained.
        offset = tf.expand_dims(tf.range(batch_size), 0)
        ancestor_inds = tf.reshape(ancestor_inds * batch_size + offset, [-1])
        noresample_inds = tf.range(self.num_samples * batch_size)
        # Decide whether or not we should resample; don't resample if we are past
        # the end of a sequence.
        should_resample = self.resampling_criterion(self.num_samples, log_ess, t)
        should_resample = tf.logical_and(should_resample,
                                     cur_mask[:batch_size] > 0.)
        float_should_resample = tf.cast(should_resample, dtype=tf.float32)
        ancestor_inds = tf.where(
                            tf.tile(should_resample, [self.num_samples]),
                            ancestor_inds,
                            noresample_inds)
        new_state = nested.gather_tensors(new_state, ancestor_inds)
        # Update the TensorArrays before we reset the weights so that we capture
        # the incremental weights and not zeros.
        new_log_weights_ta = log_weights_ta.write(t, log_weights_acc)
        # For the particle filters that resampled, update log_p_hat and
        # reset weights to zero.
    
        log_weights_acc *= (1. - tf.tile(float_should_resample[tf.newaxis, :],
                                     [self.num_samples, 1]))
        return t + 1, new_state, new_log_weights_ta, log_weights_acc

    _, _, log_weights_ta, _ = tf.while_loop(cond=while_predicate,
                                    body=while_step,
                                    loop_vars=(t0, init_states, log_weights_ta, log_weights_acc),
                                    parallel_iterations=self.parallel_iterations,
                                    swap_memory=self.swap_memory)

    log_weights = log_weights_ta.stack()
    # Add in the final weight update to log_p_hat.
    # log_weights = tf.transpose(a=log_weights, perm=[0, 2, 1])
    return log_weights