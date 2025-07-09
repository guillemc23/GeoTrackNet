
import sys

import numpy as np

sys.path.append('..')
import os
import pickle
from pathlib import Path

import tensorflow as tf

LAT, LON, SOG, COG, HEADING, ROT, NAV_STT, TIMESTAMP, MMSI = list(range(9))


# The default number of threads used to process data in parallel.
DEFAULT_PARALLELISM = 12


# In[]

def create_AIS_dataset(dataset_path,
                       mean_path,
                       batch_size,
                       data_dim,
                       lat_bins,
                       lon_bins,
                       sog_bins,
                       cog_bins,
                       num_parallel_calls=DEFAULT_PARALLELISM,
                       shuffle=True,
                       repeat=True):
    total_bins = lat_bins+lon_bins+sog_bins+cog_bins
    def sparse_AIS_to_dense(msgs_,num_timesteps, mmsis, time_start, time_end):
#        lat_bins = 200; lon_bins = 300; sog_bins = 30; cog_bins = 72
        def create_dense_vect(msg,lat_bins = 300, lon_bins = 300, sog_bins = 30 ,cog_bins = 72):
            lat, lon, sog, cog = msg[0], msg[1], msg[2], msg[3]
            data_dim = lat_bins + lon_bins + sog_bins + cog_bins
            dense_vect = np.zeros(data_dim)
            dense_vect[int(lat*lat_bins)] = 1.0
            dense_vect[int(lon*lon_bins) + lat_bins] = 1.0
            dense_vect[int(sog*sog_bins) + lat_bins + lon_bins] = 1.0
            dense_vect[int(cog*cog_bins) + lat_bins + lon_bins + sog_bins] = 1.0
            return dense_vect
        msgs_[msgs_ == 1] = 0.99999
        dense_msgs = []
        for msg in msgs_:
            # lat_bins, lon_bins, sog_bins, cog_bins are from "create_AIS_dataset" scope 
            dense_msgs.append(create_dense_vect(msg,
                                                lat_bins = lat_bins,
                                                lon_bins = lon_bins,
                                                sog_bins = sog_bins,
                                                cog_bins = cog_bins))
        dense_msgs = np.array(dense_msgs)
        return dense_msgs, num_timesteps, mmsis, time_start, time_end


    # Load the data from disk.
    with tf.io.gfile.GFile(dataset_path, "rb") as f:
        raw_data = pickle.load(f)

    num_examples = len(raw_data)
    dirname = os.path.dirname(dataset_path)

    with open(mean_path,"rb") as f:
        mean = pickle.load(f)

    def aistrack_generator():
        for k in list(raw_data.keys()):
            tmp = raw_data[k][::2,[LAT,LON,SOG,COG]] # 10 min
            tmp[tmp == 1] = 0.99999
            yield tmp, len(tmp), raw_data[k][0,MMSI], raw_data[k][0,TIMESTAMP], raw_data[k][-1,TIMESTAMP]

    dataset = tf.data.Dataset.from_generator(
                              aistrack_generator,
                              output_types=(tf.float64, tf.int64, tf.int64, tf.float32, tf.float32))
            
    if repeat: dataset = dataset.repeat()
    if shuffle: dataset = dataset.shuffle(num_examples)             
              
    dataset = dataset.map(
            lambda msg_, num_timesteps, mmsis, time_start, time_end: tuple(tf.numpy_function(sparse_AIS_to_dense,
                                                   [msg_, num_timesteps, mmsis, time_start, time_end],
                                                   [tf.float64, tf.int64, tf.int64, tf.float32, tf.float32])),
                                                num_parallel_calls=num_parallel_calls)
              

    # Batch sequences togther, padding them to a common length in time.
    dataset = dataset.padded_batch(batch_size,
                                   padded_shapes=([None, total_bins ], [], [], [], [])
                                  )


    def process_AIS_batch(data, lengths, mmsis, time_start, time_end):
        """Create mean-centered and time-major next-step prediction Tensors."""
        data = tf.cast(tf.transpose(a=data, perm=[1, 0, 2]), dtype=tf.float32)
        lengths = tf.cast(lengths, dtype=tf.int32)
        mmsis = tf.cast(mmsis, dtype=tf.int32)
        targets = data

        # Mean center the inputs.
        inputs = data - tf.constant(mean, dtype=tf.float32,
                                    shape=[1, 1, mean.shape[0]])
        # Shift the inputs one step forward in time. Also remove the last
        # timestep so that targets and inputs are the same length.
        inputs = tf.pad(tensor=data, paddings=[[1, 0], [0, 0], [0, 0]], mode="CONSTANT")[:-1]
        # Mask out unused timesteps.
        inputs *= tf.expand_dims(tf.transpose(
            a=tf.sequence_mask(lengths, dtype=inputs.dtype)), 2)
        return inputs, targets, lengths, mmsis, time_start, time_end

    dataset = dataset.map(process_AIS_batch,
                          num_parallel_calls=num_parallel_calls)


#    dataset = dataset.prefetch(num_examples)
    dataset = dataset.prefetch(50)
    inputs, targets, lengths, mmsis, time_starts, time_ends = next(iter(dataset))
    return inputs, targets, mmsis, time_starts, time_ends, lengths, tf.constant(mean, dtype=tf.float32)



def get_AIS_dataset_mean(mean_path : Path) -> tf.Tensor:
    with open(mean_path,"rb") as f:
        mean = pickle.load(f)
    return tf.constant(mean, dtype=tf.float32)

def get_Tensorflow_AIS_dataset(dataset_path,
                       batch_size,
                       lat_bins,
                       lon_bins,
                       sog_bins,
                       cog_bins,
                       num_parallel_calls=DEFAULT_PARALLELISM,
                       shuffle=True,
                       repeat=True) -> tf.data.Dataset:
    total_bins = lat_bins+lon_bins+sog_bins+cog_bins

    # Load the data from disk.
    with tf.io.gfile.GFile(dataset_path, "rb") as f:
        raw_data = pickle.load(f)

    num_examples = len(raw_data)


    def aistrack_generator():
        for k in list(raw_data.keys()):
            tmp = raw_data[k][::2,[LAT,LON,SOG,COG]] # 10 min
            tmp[tmp == 1] = 0.99999
            yield tmp, len(tmp)

    dataset = tf.data.Dataset.from_generator(
                              aistrack_generator,
                              output_types=(tf.float64, tf.int32))
            
    if repeat: dataset = dataset.repeat()
    if shuffle: dataset = dataset.shuffle(num_examples)             
              

    def sparse_AIS_to_dense(msgs_,length):
#        lat_bins = 200; lon_bins = 300; sog_bins = 30; cog_bins = 72
        def create_dense_vect(msg,lat_bins = 300, lon_bins = 300, sog_bins = 30 ,cog_bins = 72):
            lat, lon, sog, cog = msg[0], msg[1], msg[2], msg[3]
            data_dim = lat_bins + lon_bins + sog_bins + cog_bins
            dense_vect = np.zeros(data_dim)
            dense_vect[int(lat*lat_bins)] = 1.0
            dense_vect[int(lon*lon_bins) + lat_bins] = 1.0
            dense_vect[int(sog*sog_bins) + lat_bins + lon_bins] = 1.0
            dense_vect[int(cog*cog_bins) + lat_bins + lon_bins + sog_bins] = 1.0
            return dense_vect
        msgs_[msgs_ == 1] = 0.99999
        dense_msgs = []
        for msg in msgs_:
            # lat_bins, lon_bins, sog_bins, cog_bins are from "create_AIS_dataset" scope 
            dense_msgs.append(create_dense_vect(msg,
                                                lat_bins = lat_bins,
                                                lon_bins = lon_bins,
                                                sog_bins = sog_bins,
                                                cog_bins = cog_bins))
        dense_msgs = np.array(dense_msgs, dtype=np.float32)
        return dense_msgs, length.astype(np.int32)


    dataset = dataset.map(
            lambda msg_, lengths: tuple(tf.numpy_function(sparse_AIS_to_dense,
                                                   [msg_, lengths],
                                                   [tf.float32, tf.int32])),
                                                num_parallel_calls=num_parallel_calls)
              

    # Batch sequences togther, padding them to a common length in time.
    dataset = dataset.padded_batch(batch_size,
                                   padded_shapes=([None, total_bins ], [])
                                  )


    def process_AIS_batch(data, lengths):
        """Create mean-centered and time-major next-step prediction Tensors."""
        data = tf.transpose(a=data, perm=[1, 0, 2])
        targets = data
        # Shift the inputs one step forward in time. Also remove the last
        # timestep so that targets and inputs are the same length.
        inputs = tf.pad(tensor=data, paddings=[[1, 0], [0, 0], [0, 0]], mode="CONSTANT")[:-1]
        # Mask out unused timesteps.
        inputs *= tf.expand_dims(tf.transpose(
            a=tf.sequence_mask(lengths, dtype=inputs.dtype)), 2)
        return inputs, targets, lengths

    dataset = dataset.map(process_AIS_batch,
                          num_parallel_calls=num_parallel_calls)


#    dataset = dataset.prefetch(num_examples)
    dataset = dataset.prefetch(50)
    return dataset



def get_Tensorflow_AIS_dataset_uint(dataset_path,
                       batch_size,
                       lat_bins,
                       lon_bins,
                       sog_bins,
                       cog_bins,
                       num_parallel_calls=DEFAULT_PARALLELISM,
                       shuffle=True,
                       repeat=True) -> tf.data.Dataset:
    total_bins = lat_bins+lon_bins+sog_bins+cog_bins

    # Load the data from disk.
    with tf.io.gfile.GFile(dataset_path, "rb") as f:
        raw_data = pickle.load(f)

    num_examples = len(raw_data)


    def aistrack_generator():
        for k in list(raw_data.keys()):
            tmp = raw_data[k][::2,[LAT,LON,SOG,COG]] # 10 min
            tmp[tmp == 1] = 0.99999
            yield tmp, len(tmp)

    dataset = tf.data.Dataset.from_generator(
                              aistrack_generator,
                              output_types=(tf.float64, tf.int32))
            
    if repeat: dataset = dataset.repeat()
    if shuffle: dataset = dataset.shuffle(num_examples)             
              

    def sparse_AIS_to_dense(msgs_,length):
#        lat_bins = 200; lon_bins = 300; sog_bins = 30; cog_bins = 72
        def create_dense_vect(msg,lat_bins = 300, lon_bins = 300, sog_bins = 30 ,cog_bins = 72):
            lat, lon, sog, cog = msg[0], msg[1], msg[2], msg[3]
            data_dim = lat_bins + lon_bins + sog_bins + cog_bins
            dense_vect = np.zeros(data_dim, np.bool)
            dense_vect[int(lat*lat_bins)] = True
            dense_vect[int(lon*lon_bins) + lat_bins] = True
            dense_vect[int(sog*sog_bins) + lat_bins + lon_bins] = True
            dense_vect[int(cog*cog_bins) + lat_bins + lon_bins + sog_bins] = True
            return dense_vect
        msgs_[msgs_ == 1] = 0.99999
        dense_msgs = []
        for msg in msgs_:
            # lat_bins, lon_bins, sog_bins, cog_bins are from "create_AIS_dataset" scope 
            dense_msgs.append(create_dense_vect(msg,
                                                lat_bins = lat_bins,
                                                lon_bins = lon_bins,
                                                sog_bins = sog_bins,
                                                cog_bins = cog_bins))
        dense_msgs = np.array(dense_msgs, dtype=np.uint8)
        return dense_msgs, length.astype(np.int32)


    dataset = dataset.map(
            lambda msg_, lengths: tuple(tf.numpy_function(sparse_AIS_to_dense,
                                                   [msg_, lengths],
                                                   [tf.uint8, tf.int32])),
                                                num_parallel_calls=num_parallel_calls)
              

    # Batch sequences togther, padding them to a common length in time.
    dataset = dataset.padded_batch(batch_size,
                                   padded_shapes=([None, total_bins ], [])
                                  )


    def process_AIS_batch(data, lengths):
        """Create mean-centered and time-major next-step prediction Tensors."""
        data = tf.transpose(a=data, perm=[1, 0, 2])
        targets = data
        # Shift the inputs one step forward in time. Also remove the last
        # timestep so that targets and inputs are the same length.
        inputs = tf.pad(tensor=data, paddings=[[1, 0], [0, 0], [0, 0]], mode="CONSTANT")[:-1]
        # Mask out unused timesteps.
        inputs *= tf.expand_dims(tf.transpose(
            a=tf.sequence_mask(lengths, dtype=inputs.dtype)), 2)
        return tf.cast(inputs, dtype=tf.float32), tf.cast(targets, dtype=tf.float32), lengths

    dataset = dataset.map(process_AIS_batch,
                          num_parallel_calls=num_parallel_calls)


#    dataset = dataset.prefetch(num_examples)
    dataset = dataset.prefetch(50)
    return dataset

