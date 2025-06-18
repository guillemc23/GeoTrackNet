import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import layers


class ConditionalNormalDistribution(layers.Layer):
  """A Normal distribution conditioned on Tensor inputs via a fc network."""

  def __init__(self, size : int, hidden_layer_size : int, sigma_min : int =0.0,
               raw_sigma_bias=0.25):
    """Creates a conditional Normal distribution.

    Args:
      size: The dimension of the random variable.
      hidden_layers_size: The size of the hidden layers of the fully connected
        network used to condition the distribution on the inputs.
      sigma_min: The minimum standard deviation allowed, a scalar.
      raw_sigma_bias: A scalar that is added to the raw standard deviation
        output from the fully connected network. Set to 0.25 by default to
        prevent standard deviations close to 0.
    """
    super().__init__()
    
    # Trainable parameters
    self.dense1 = layers.Dense(hidden_layer_size, activation="relu")
    self.dense2 = layers.Dense(2*size, activation=None)

    # static parameters
    self.sigma_min = sigma_min
    self.raw_sigma_bias = raw_sigma_bias



  def _estimate_mu_and_sigma(self, inputs : tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    """Computes the parameters of a normal distribution based on the inputs."""

    # predict mu and sigma
    outs = self.dense1(inputs)
    outs = self.dense2(outs)
    mu, sigma = tf.split(outs, 2, axis=1)
    # correct
    sigma = tf.maximum(tf.nn.softplus(sigma + self.raw_sigma_bias), self.sigma_min)
    return mu, sigma
  
  def call(self, inputs : tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    """Creates a normal distribution conditioned on the inputs."""
    mu, sigma = self._estimate_mu_and_sigma(inputs)
    return tfp.distributions.Normal(loc=mu, scale=sigma)
  

class NormalApproximatePosterior(ConditionalNormalDistribution):
  """A Normally-distributed approx. posterior with res_q parameterization."""

  def _estimate_mu_and_sigma(self, inputs : tf.Tensor, prior_mu : int) -> tuple[tf.Tensor, tf.Tensor]: 
    """Generates the mean and variance of the normal distribution.

    Args:
      tensor_list: The list of Tensors to condition on. Will be concatenated and
        fed through a fully connected network.
      prior_mu: The mean of the prior distribution associated with this
        approximate posterior. Will be added to the mean produced by
        this approximate posterior, in res_q fashion.
    Returns:
      mu: The mean of the approximate posterior.
      sigma: The standard deviation of the approximate posterior.
    """

    mu, sigma = super(NormalApproximatePosterior, self)._estimate_mu_and_sigma(inputs)
    return mu + prior_mu, sigma

class ConditionalBernoulliDistribution(layers.Layer):
  """A Normal distribution conditioned on Tensor inputs via a fc network."""

  def __init__(self, size : int, hidden_layer_size : int, bias_init : float = 0.0):
    """Creates a conditional Normal distribution.

    Args:
      size: The dimension of the random variable.
      hidden_layers_size: The size of the hidden layers of the fully connected
        network used to condition the distribution on the inputs.
      sigma_min: The minimum standard deviation allowed, a scalar.
      raw_sigma_bias: A scalar that is added to the raw standard deviation
        output from the fully connected network. Set to 0.25 by default to
        prevent standard deviations close to 0.
    """
    super().__init__()

    self.bias_init = bias_init

    # Trainable parameters
    self.dense1 = layers.Dense(hidden_layer_size, activation="relu")
    self.dense2 = layers.Dense(size, activation=None)

    # static parameters
    self.bias_init = bias_init

  def _estimate_p(self, inputs : tf.Tensor) -> tf.Tensor:
    """Computes the p parameter of the Bernoulli distribution."""

    outs = self.dense1(inputs)
    outs = self.dense2(outs)
    return outs + self.bias_init
  
  def call(self, inputs : tf.Tensor):
    """Creates a normal distribution conditioned on the inputs."""
    # inputs = tf.concat(tensor_list, axis=1)
    p = self._estimate_p(inputs)
    return tfp.distributions.Bernoulli(logits=p)