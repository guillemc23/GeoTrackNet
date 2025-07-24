from typing import Callable

import tensorflow as tf
from pydantic import BaseModel, Field, PositiveInt, computed_field

# class TensorflowEncodedBatchedDatasetWrapper:
#     def __init__(self, dataset : tf.data.Dataset, generator_data_size : PositiveInt, ):
#         self._dataset = dataset

#     def __iter__(self):
#         return self._dataset
    
#     def __len__(self):


class TensorflowEncodedBatchedDatasetBuilder(BaseModel):
    track_generator : Callable
    batch_size : PositiveInt
    lat_bins : PositiveInt
    lon_bins : PositiveInt
    sog_bins : PositiveInt
    cog_bins : PositiveInt
    num_parallel_calls : int = Field(tf.data.AUTOTUNE)
    repeat : bool = Field(True)
    shuffle : bool = Field(True)
    
    @computed_field
    @property
    def total_bins(self) -> PositiveInt:
        return self.lat_bins+self.lon_bins+self.sog_bins+self.cog_bins

    def _one_hot_encode(self, dataset : tf.data.Dataset) -> tf.data.Dataset:
        bins = tf.constant([self.lat_bins, self.lon_bins, self.sog_bins, self.cog_bins], dtype = tf.float32)
        bins_cumsum_shifted = tf.cast(tf.pad(tf.cumsum(bins)[:-1], paddings=[[1,0]]), dtype=tf.int32)

        @tf.function(
            input_signature=(
                tf.TensorSpec(shape=[None,4], dtype=tf.float32,name='msgs'),
            )
        )
        def _one_hot_encode_track_messages_vec(msgs : tf.Tensor) -> tf.Tensor:
            enc_msgs = tf.reduce_sum(tf.one_hot(bins_cumsum_shifted + tf.cast(msgs * bins, dtype=tf.int32), depth=self.total_bins), axis=1)
            return enc_msgs, len(enc_msgs)
        
        dataset = dataset.map(_one_hot_encode_track_messages_vec, 
                              num_parallel_calls=self.num_parallel_calls,
                               deterministic=False)
        return dataset

    def _add_shifted_forward_msg_attribute(self, dataset : tf.data.Dataset) -> tf.data.Dataset:
        
        def _add_shifted_forward_msg_attribute_to_row(msg, lengths):
            """Create mean-centered and time-major next-step prediction Tensors."""
            msg = tf.transpose(a=msg, perm=[1, 0, 2])
            msg_copy = msg
            # Shift the inputs one step forward in time. Also remove the last
            # timestep so that targets and inputs are the same length.
            msg_shift_forward = tf.pad(tensor=msg, paddings=[[1, 0], [0, 0], [0, 0]], mode="CONSTANT")[:-1]
            # Mask out unused timesteps.
            msg_shift_forward *= tf.expand_dims(tf.transpose(
                a=tf.sequence_mask(lengths, dtype=msg_shift_forward.dtype)), 2)
            return msg_shift_forward, msg_copy, lengths

        dataset = dataset.map(_add_shifted_forward_msg_attribute_to_row,
                            num_parallel_calls=self.num_parallel_calls, 
                            deterministic=False)
        return dataset
    
    # def get_dataset_length_after_build(self):


    
    def build(self)-> tf.data.Dataset:
        num_tracks = 105_325
        # len(self.track_generator)
        dataset = tf.data.Dataset.from_generator(
                            self.track_generator,
                            output_signature=(
                              tf.TensorSpec(shape=(None, 4), dtype=tf.float32))
                 )
        if self.repeat: 
            dataset = dataset.repeat()
        if self.shuffle: 
            dataset = dataset.shuffle(num_tracks)
        
        dataset = self._one_hot_encode(dataset)
        # Batch sequences togther, padding them to a common length in time.
        dataset = dataset.padded_batch(self.batch_size,
                                   padded_shapes=([None, self.total_bins ], []),
                                   drop_remainder=True)
        
        dataset = self._add_shifted_forward_msg_attribute(dataset)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset
    



class TensorflowEncodedBatchedInferenceDatasetBuilder(BaseModel):
    track_with_id_generator : Callable
    batch_size : PositiveInt
    lat_bins : PositiveInt
    lon_bins : PositiveInt
    sog_bins : PositiveInt
    cog_bins : PositiveInt
    num_parallel_calls : int = Field(tf.data.AUTOTUNE)
    repeat : bool = Field(True)
    shuffle : bool = Field(True)
    
    @computed_field
    @property
    def total_bins(self) -> PositiveInt:
        return self.lat_bins+self.lon_bins+self.sog_bins+self.cog_bins

    def _one_hot_encode(self, dataset : tf.data.Dataset) -> tf.data.Dataset:
        bins = tf.constant([self.lat_bins, self.lon_bins, self.sog_bins, self.cog_bins], dtype = tf.float32)
        bins_cumsum_shifted = tf.cast(tf.pad(tf.cumsum(bins)[:-1], paddings=[[1,0]]), dtype=tf.int32)

        @tf.function(
            input_signature=(
                tf.TensorSpec(shape=(), dtype=tf.uint16,name='track_id'),
                tf.TensorSpec(shape=(None,4), dtype=tf.float32,name='msgs'),
            )
        )
        def _one_hot_encode_track_messages_vec(track_id : tf.Tensor, msgs : tf.Tensor) -> tf.Tensor:
            enc_msgs = tf.reduce_sum(tf.one_hot(bins_cumsum_shifted + tf.cast(msgs * bins, dtype=tf.int32), depth=self.total_bins), axis=1)
            return track_id, enc_msgs, len(enc_msgs)
        
        dataset = dataset.map(_one_hot_encode_track_messages_vec, 
                              num_parallel_calls=self.num_parallel_calls,
                               deterministic=False)
        return dataset

    def _add_shifted_forward_msg_attribute(self, dataset : tf.data.Dataset) -> tf.data.Dataset:
        
        def _add_shifted_forward_msg_attribute_to_row(track_ids, msg, lengths):
            """Create mean-centered and time-major next-step prediction Tensors."""
            msg = tf.transpose(a=msg, perm=[1, 0, 2])
            msg_copy = msg
            # Shift the inputs one step forward in time. Also remove the last
            # timestep so that targets and inputs are the same length.
            msg_shift_forward = tf.pad(tensor=msg, paddings=[[1, 0], [0, 0], [0, 0]], mode="CONSTANT")[:-1]
            # Mask out unused timesteps.
            msg_shift_forward *= tf.expand_dims(tf.transpose(
                a=tf.sequence_mask(lengths, dtype=msg_shift_forward.dtype)), 2)
            return track_ids, msg_shift_forward, msg_copy, lengths

        dataset = dataset.map(_add_shifted_forward_msg_attribute_to_row,
                            num_parallel_calls=self.num_parallel_calls, 
                            deterministic=False)
        return dataset
    
    # def get_dataset_length_after_build(self):


    
    def build(self)-> tf.data.Dataset:
        num_tracks = 105_325
        # len(self.track_generator)
        dataset = tf.data.Dataset.from_generator(
                            self.track_with_id_generator,
                            output_signature=(
                              tf.TensorSpec(shape=(), dtype=tf.uint16, name='track_id'),
                              tf.TensorSpec(shape=(None, 4), dtype=tf.float32, name='msgs'))
                              
                 )
        if self.repeat: 
            dataset = dataset.repeat()
        if self.shuffle: 
            dataset = dataset.shuffle(num_tracks)
        
        dataset = self._one_hot_encode(dataset)
        # Batch sequences togther, padding them to a common length in time.
        dataset = dataset.padded_batch(self.batch_size,
                                   padded_shapes=([], [None, self.total_bins ], []),
                                   drop_remainder=True)
        
        dataset = self._add_shifted_forward_msg_attribute(dataset)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset