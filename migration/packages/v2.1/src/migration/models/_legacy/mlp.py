
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


def get_mlp(layer_sizes: tuple[int, int], activate_final : bool = True):
  model = Sequential(
    [
        layers.Dense(layer_sizes[0], activation="relu"),
        layers.Dense(layer_sizes[1], activation="relu" if activate_final else None),
    ])
  return model


