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
               name="vrnn_cell"):
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
    return (self.rnn_cell.get_initial_state(batch_size),
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
    latent_dist_prior = self.prior(rnn_out)  
    latent_dist_q_input = tf.concat([rnn_out, targets_encoded], axis=1)                                   # STEP 1: p(z_t|h_t) = phi_prior(o_t) = p(z_t|x_<t, z_<t)
    latent_dist_q = self.approx_posterior(latent_dist_q_input,             # STEP 4: q(z_t|x_t) = phi_enc(o_t, phi_x(x_t))
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
          
