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

from __future__ import absolute_import, division, print_function

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import Layer, Model

from migration.models.distributions import (
  ConditionalBernoulliDistribution,
  ConditionalNormalDistribution,
  NormalApproximatePosterior,
)
from migration.models.legacy.mlp import get_mlp


class VRNNCell(Model):

  def __init__(self,
               rnn_cell : Layer,
               data_feat_extractor : Layer,
               latent_feat_extractor : Layer,
               latent_feat_extractor_output_size : int,
               prior : Layer,
               approx_posterior : Layer,
               generative : Layer,
               random_seed=None,
               name="vrnn"):
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
    super(VRNNCell, self).__init__(name=name)
    self.rnn_cell = rnn_cell
    self.data_feat_extractor = data_feat_extractor
    self.latent_feat_extractor = latent_feat_extractor
    self.prior = prior
    self.approx_posterior = approx_posterior
    self.generative = generative
    self.random_seed = random_seed
    self.encoded_z_size = latent_feat_extractor_output_size
    self.state_size = (self.rnn_cell.state_size, self.encoded_z_size)

  def zero_state(self, batch_size, dtype):
    """The initial state of the VRNN.

    Contains the initial state of the RNN as well as a vector of zeros
    corresponding to z_0.
    Args:
      batch_size: The batch size.
      dtype: The data type of the VRNN.
    Returns:
      zero_state: The initial state of the VRNN.
    """
    return (self.rnn_cell.zero_state(batch_size, dtype),
            tf.zeros([batch_size, self.encoded_z_size], dtype=dtype))

  def call(self, observations, state, mask, return_value = None):
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
    inputs, targets = observations                                              # x_t-1, x_t
    no_target = tf.equal(tf.reduce_sum(input_tensor=targets),0)
    rnn_state, prev_latent_encoded = state                                      # h_t-1, phi_z(z_t-1)
    # Encode the data.
    inputs_encoded = self.data_feat_extractor(inputs)                           # phi_x(x_t-1)
    targets_encoded = self.data_feat_extractor(targets)                         # phi_x(x_t)
    # Run the RNN cell.
    rnn_inputs = tf.concat([inputs_encoded, prev_latent_encoded], axis=1)
    rnn_out, new_rnn_state = self.rnn_cell(rnn_inputs, rnn_state)               # STEP 3: o_t, h_t = RNN(phi_x(x_t-1), phi_z(z_t-1), h_t-1)
    # Create the prior and approximate posterior distributions.
    latent_dist_prior = self.prior(rnn_out)                                     # STEP 1: p(z_t|h_t) = phi_prior(o_t) = p(z_t|x_<t, z_<t)
    latent_dist_q = self.approx_posterior(rnn_out, targets_encoded,             # STEP 4: q(z_t|x_t) = phi_enc(o_t, phi_x(x_t))
                                          prior_mu=latent_dist_prior.loc)
    # Sample the new latent state z and encode it.
    latent_state = latent_dist_q.sample(seed=self.random_seed)                  # z_t ~ q(z_t|x_t) = phi_enc(o_t, phi_x(x_t))
    latent_encoded = self.latent_feat_extractor(latent_state)                   # phi_z(z_t)
    
    latent_state_prior = latent_dist_prior.sample(seed=self.random_seed)                  # z_t ~ q(z_t|x_t) = phi_enc(o_t, phi_x(x_t))    
    latent_prior_encoded = self.latent_feat_extractor(latent_state_prior)                   # phi_z(z_t)
    
    # Calculate probabilities of the latent state according to the prior p
    # and approximate posterior q.
    log_q_z = tf.reduce_sum(input_tensor=latent_dist_q.log_prob(latent_state), axis=-1)
    log_p_z = tf.reduce_sum(input_tensor=latent_dist_prior.log_prob(latent_state), axis=-1)
    analytic_kl = tf.reduce_sum(
        input_tensor=tfp.distributions.kl_divergence(
            latent_dist_q, latent_dist_prior),
        axis=-1)
    # Create the generative dist. and calculate the logprob of the targets.
    # TODO: moved concat to here
    generative_dist = self.generative(tf.concat([latent_encoded, rnn_out], axis=1))                  # STEP 2: p(x_t|z_t) = phi_dec(z_t, o_t) = p(x_t|z_<=t, x<t)
    log_p_x_given_z = tf.reduce_sum(input_tensor=generative_dist.log_prob(targets), axis=-1)
    
    # TODO: moved concat to here
    generative_dist_prior = self.generative(tf.concat([latent_prior_encoded, rnn_out], axis=1))                  # STEP 2: p(x_t|z_t) = phi_dec(z_t, o_t) = p(x_t|z_<=t, x<t)
    
    latent_encoded_return = tf.cond(pred=no_target, true_fn=lambda: latent_prior_encoded, false_fn=lambda: latent_encoded)
    
    if return_value is None:
        return (log_q_z, log_p_z, log_p_x_given_z, analytic_kl,
                (new_rnn_state, latent_encoded_return), rnn_out)
    
    if return_value == "logits":
        dists_return = tf.cond(pred=no_target, 
                               true_fn=lambda: generative_dist_prior.logits, 
                               false_fn=lambda: generative_dist.logits)
    elif return_value == "probs":
        dists_return = tf.cond(pred=no_target, 
                               true_fn=lambda: generative_dist_prior.probs, 
                               false_fn=lambda: generative_dist.probs)
    
    return (log_q_z, log_p_z, log_p_x_given_z, analytic_kl,
            (new_rnn_state, latent_encoded_return), rnn_out, dists_return)
            
  def sample(self,observations, state):
    with tf.compat.v1.variable_scope("vrnn"): 
        inputs = observations                                              # x_t-1, x_t
        rnn_state, prev_latent_encoded = state                                      # h_t-1, phi_z(z_t-1)
        # Encode the data.
        inputs_encoded = self.data_feat_extractor(inputs)                           # phi_x(x_t-1)
        # Run the RNN cell.
        rnn_inputs = tf.concat([inputs_encoded, prev_latent_encoded], axis=1)
        rnn_out, new_rnn_state = self.rnn_cell(rnn_inputs, rnn_state)               # STEP 3: o_t, h_t = RNN(phi_x(x_t-1), phi_z(z_t-1), h_t-1)
        # Create the prior and approximate posterior distributions.
        latent_dist_prior = self.prior(rnn_out)                                     # STEP 1: p(z_t|h_t) = phi_prior(o_t) = p(z_t|x_<t, z_<t)
        # Sample the new latent state z and encode it.
        latent_state_prior = latent_dist_prior.sample(seed=self.random_seed)        # z_t ~ p(z_t|h_t)
        latent_prior_encoded = self.latent_feat_extractor(latent_state_prior)                   # phi_z(z_t)
        # Calculate probabilities of the latent state according to the prior p
        # and approximate posterior q.
        # Create the generative dist. and calculate the logprob of the targets.
        generative_dist = self.generative(latent_prior_encoded, rnn_out)                  # STEP 2: p(x_t|z_t) = phi_dec(z_t, o_t) = p(x_t|z_<=t, x<t)
#        new_sample = generative_dist.sample(seed=self.random_seed)
#        new_sample = tf.cast(new_sample,tf.float32)
        logits_return = generative_dist.logits
    return (logits_return, (new_rnn_state, latent_prior_encoded))


_DEFAULT_INITIALIZERS = {"w": tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"),
                         "b": tf.compat.v1.zeros_initializer()}


def create_vrnn(
    data_size : int,
    latent_size : int,
    generative_class,
    sigma_min=0.0,
    raw_sigma_bias=0.25,
    generative_bias_init=0.0,
    initializers=None,
    random_seed=None):
  """A factory method for creating VRNN cells.

  Args:
    data_size: The dimension of the vectors that make up the data sequences.
    latent_size: The size of the stochastic latent state of the VRNN.
    generative_class: The class of the generative distribution. Can be either
      ConditionalNormalDistribution or ConditionalBernoulliDistribution.
    rnn_hidden_size: The hidden state dimension of the RNN that forms the
      deterministic part of this VRNN. If None, then it defaults
      to latent_size.
    fcnet_hidden_sizes: A list of python integers, the size of the hidden
      layers of the fully connected networks that parameterize the conditional
      distributions of the VRNN. If None, then it defaults to one hidden
      layer of size latent_size.
    encoded_data_size: The size of the output of the data encoding network. If
      None, defaults to latent_size.
    encoded_latent_size: The size of the output of the latent state encoding
      network. If None, defaults to latent_size.
    sigma_min: The minimum value that the standard deviation of the
      distribution over the latent state can take.
    raw_sigma_bias: A scalar that is added to the raw standard deviation
      output from the neural networks that parameterize the prior and
      approximate posterior. Useful for preventing standard deviations close
      to zero.
    generative_bias_init: A bias to added to the raw output of the fully
      connected network that parameterizes the generative distribution. Useful
      for initalizing the mean of the distribution to a sensible starting point
      such as the mean of the training data. Only used with Bernoulli generative
      distributions.
    initializers: The variable intitializers to use for the fully connected
      networks and RNN cell. Must be a dictionary mapping the keys 'w' and 'b'
      to the initializers for the weights and biases. Defaults to xavier for
      the weights and zeros for the biases when initializers is None.
    random_seed: A random seed for the VRNN resampling operations.
  Returns:
    model: A VRNNCell object.
  """

  if initializers is None:
    # TODO: review initializers
    initializers = _DEFAULT_INITIALIZERS


  data_feat_extractor = get_mlp(
    layer_sizes=[latent_size, latent_size],

  )
  latent_feat_extractor = get_mlp(
    layer_sizes=[latent_size, latent_size],

  )
  prior = ConditionalNormalDistribution(
      size=latent_size,
      hidden_layer_sizes=[latent_size],
      sigma_min=sigma_min,
      raw_sigma_bias=raw_sigma_bias,
      initializers=initializers,
      name="prior")
  approx_posterior = NormalApproximatePosterior(
      size=latent_size,
      hidden_layer_sizes=[latent_size],
      sigma_min=sigma_min,
      raw_sigma_bias=raw_sigma_bias,
      initializers=initializers,
      name="approximate_posterior")
  if generative_class == ConditionalBernoulliDistribution:
    generative = ConditionalBernoulliDistribution(
        size=data_size,
        hidden_layer_sizes=[latent_size],
        initializers=initializers,
        bias_init=generative_bias_init,
        name="generative")
  else:
    generative = ConditionalNormalDistribution(
        size=data_size,
        hidden_layer_sizes=[latent_size],
        initializers=initializers,
        name="generative")
  # weight initializer maybe?
  # TODO: review
  rnn_layer = tf.keras.layers.LSTMCell(latent_size)
  return VRNNCell(rnn_layer, data_feat_extractor, latent_feat_extractor, latent_size,
                  prior, approx_posterior, generative, random_seed=random_seed)
