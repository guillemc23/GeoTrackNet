
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


def get_mlp(layer_sizes: tuple[int, int], initializers : dict, activate_final : bool = True, ):
  model = Sequential(
    [
        layers.Dense(layer_sizes[0], activation="relu", kernel_initializer=initializers['w'], bias_initializer=initializers['b']),
        layers.Dense(layer_sizes[1], activation="relu" if activate_final else None, kernel_initializer=initializers['w'], bias_initializer=initializers['b']),
    ])
  return model


