
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


def build_mlp(layer_sizes: tuple[int, int], initializers : dict, activate_final : bool = True, name : str= None):
  model = Sequential(
    [
        layers.Dense(layer_sizes[0], activation="relu", kernel_initializer=initializers['w'], bias_initializer=initializers['b']),
        layers.Dense(layer_sizes[1], activation="relu" if activate_final else None, kernel_initializer=initializers['w'], bias_initializer=initializers['b']),
    ], name=name)
  return model


