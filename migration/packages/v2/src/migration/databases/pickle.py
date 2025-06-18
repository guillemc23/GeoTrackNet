import pickle
from pathlib import Path

import numpy as np
import tensorflow as tf

from migration.dataset.datasets import AISDatabase, DiscreteInterval

LAT_IDX, LON_IDX, SOG_IDX, COG_IDX, HEADING_IDX, ROT_IDX, NAV_STT_IDX, TIMESTAMP_IDX, MMSI_IDX = list(range(9))

class LegacyPickle(AISDatabase):
        def __init__(self, dataset_path : Path, correct_outliers : bool = True):
            dataset_path = Path(dataset_path)
            assert dataset_path.exists(), 'Dataset path does not exist'
            self.dataset_path = dataset_path

        def get_hot_encoded_dataset(self, time_sampling_min : int, lat : DiscreteInterval, lon : DiscreteInterval, sog : DiscreteInterval, cog : DiscreteInterval, mean_scaling : bool) -> tf.data.Dataset:

            self.lat_bins = lat.num_bins
            self.lon_bins = lon.num_bins
            self.cog_bins = cog.num_bins
            self.sog_bins = sog.num_bins
            self.total_bins = self.lat_bins + self.lon_bins + self.sog_bins + self.cog_bins

            with tf.io.gfile.GFile(self.dataset_path, "rb") as f:
                raw_data = pickle.load(f)
        
            def aistrack_generator():
                for k in list(raw_data.keys()):
                    input_data = raw_data[k][::2,[LAT_IDX,LON_IDX,SOG_IDX,COG_IDX]] # 10 min
                    # snap out of bounds bins
                    input_data[input_data == 1] = 0.99999
                    yield input_data

            dataset = tf.data.Dataset.from_generator(
                        aistrack_generator,
                        output_types=(tf.float64))
            
            dataset = self._one_hot_encode_dataset(dataset)
            if mean_scaling:
                dataset_mean : tf.Tensor = self._compute_dataset_mean(dataset)
                dataset = dataset.map(lambda x : x - dataset_mean)
            return dataset

    
        def _one_hot_encode_dataset(self, ds : tf.data.Dataset)-> tf.data.Dataset:
            # TODO: maybe int and not float64 lol
            @tf.numpy_function(Tout=tf.float64)
            def msg_to_4_hot_encoding(msg : np.ndarray) -> np.ndarray:
                "Expects a 9-dimensional message"
                lat, lon, sog, cog = msg[LAT_IDX], msg[LON_IDX], msg[SOG_IDX], msg[COG_IDX]
                dense_vect = np.zeros(self.total_bins)
                dense_vect[int(lat*self.lat_bins)] = 1.0
                dense_vect[int(lon*self.lon_bins) + self.lat_bins] = 1.0
                dense_vect[int(sog*self.sog_bins) + self.lat_bins + self.lon_bins] = 1.0
                dense_vect[int(cog*self.cog_bins) + self.lat_bins + self.lon_bins + self.sog_bins] = 1.0
                return dense_vect
            dataset = ds.map(lambda x: tf.map_fn(fn=msg_to_4_hot_encoding, elems=x))
            return dataset
        
        def _compute_dataset_mean(self, ds : tf.data.Dataset) -> tf.Tensor:
            mean_per_mmsi = ds.map(lambda x: tf.math.reduce_mean(x, axis=0))
            dataset_mean = mean_per_mmsi.reduce(np.float64(0), lambda x,y: x+y) / mean_per_mmsi.reduce(np.float64(0), lambda x, _: x+1)
            return dataset_mean
