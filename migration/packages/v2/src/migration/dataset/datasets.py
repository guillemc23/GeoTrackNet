# coding: utf-8

# MIT License
# 
# Copyright (c) 2018 Duong Nguyen
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ==============================================================================

"""
Input pipelines script for Tensorflow graph.
This script is adapted from the original script of FIVO.
"""

from abc import ABC, abstractmethod
from pathlib import Path

import tensorflow as tf

# The default number of threads used to process data in parallel.
DEFAULT_PARALLELISM = 12

class DiscreteInterval:
    min_val : float 
    max_val : float 
    num_bins : int
    def __init__(self, min_val : float, max_val : float, num_bins : int):
        self.min_val = min_val
        self.max_val = max_val
        self.num_bins = num_bins



# FIXME: deprecated
def open_dataset_from_pickle_dict(pickle_path : Path, correct_outliers: bool = True):
    "dataset where each entry is a (Tx, 4) tensor, where Tx is the number of messages of MMSI x"



class AISDatabase(ABC):

    @abstractmethod
    def get_hot_encoded_dataset(self, time_sampling_min : int, lat : DiscreteInterval, lon : DiscreteInterval, sog : DiscreteInterval, cog : DiscreteInterval, mean_scaling : bool) -> tf.data.Dataset:
        """Returns a tensorflow dataset that has each element as a tensor of shape (x, lat_bins+lon_bins+ sog_bins+cog_bins)

        Returns:
            tf.data.Dataset: _description_
        """        
        pass


class DatasetBuilder:
    def __init__(self, ais_db : AISDatabase,
                       batch_size : int,
                       lat_bins : int,
                       lon_bins : int,
                       sog_bins : int ,
                       cog_bins : int,
                       num_parallel_calls : int =DEFAULT_PARALLELISM,
                       shuffle_size : int = 1,
                       repeat_indefinitely : bool =True):
        self.ais_db = ais_db
        self.batch_size = batch_size
        self.lat_bins = lat_bins
        self.lon_bins = lon_bins
        self.sog_bins = sog_bins
        self.cog_bins = cog_bins
        self.total_bins = lat_bins + lon_bins + sog_bins + cog_bins
        self.num_parallel_calls = num_parallel_calls
        self.shuffle_size = shuffle_size
        self.repeat_indefinitely = repeat_indefinitely
    

    def build(self, seed : int = 0) -> tf.data.Dataset:
        """Returns a Dataset with each row being a tuple of tensors: the input and target, both of them of shape (T_row, batch_size, total_bins)
        """        

        targets = self.ais_db.get_hot_encoded_dataset(
            10, 
            DiscreteInterval(0,0, self.lat_bins),
            DiscreteInterval(0,0, self.lon_bins),
            DiscreteInterval(0,0, self.sog_bins),
            DiscreteInterval(0,0, self.cog_bins), mean_scaling=False)

        inputs = self.ais_db.get_hot_encoded_dataset(
            10, 
            DiscreteInterval(0,0, self.lat_bins),
            DiscreteInterval(0,0, self.lon_bins),
            DiscreteInterval(0,0, self.sog_bins),
            DiscreteInterval(0,0, self.cog_bins), mean_scaling=False)
        
        if self.repeat_indefinitely: 
            # NOTE: I think this is no longer needed since this is done to iterate it over more than one epoch
            targets = targets.repeat()
            inputs = inputs.repeat()
        if self.shuffle_size > 1:
            # NOTE: Before it shuffled the entire dataset, which has to load everything in memory
            targets = targets.shuffle(self.shuffle_size, seed=seed)  
            inputs = inputs.shuffle(self.shuffle_size, seed=seed)  
        # Change shape to (self.batch_size, T, self.total_bins), where T is the max amount of messages of the members of batch
        inputs = inputs.padded_batch(self.batch_size, padded_shapes=([None, self.total_bins]))
        targets = targets.padded_batch(self.batch_size, padded_shapes=([None, self.total_bins]))

        # reshape (T, self.batch_size, self.total_bins)
        inputs = inputs.map(lambda x: tf.cast(tf.transpose(a=x, perm=[1,0,2]), dtype=tf.float32))
        targets = targets.map(lambda x: tf.cast(tf.transpose(a=x, perm=[1,0,2]), dtype=tf.float32))

        # Shift the inputs one step forward in time. Also remove the last
        # timestep so that targets and inputs are the same length.
        inputs = inputs.map(lambda x: tf.pad(tensor=x, paddings=[[1, 0], [0, 0], [0, 0]], mode="CONSTANT")[:-1])
    
        dataset = tf.data.Dataset.zip(inputs, targets)
    #    dataset = dataset.prefetch(num_examples)
        dataset = dataset.prefetch(50)
        return dataset


