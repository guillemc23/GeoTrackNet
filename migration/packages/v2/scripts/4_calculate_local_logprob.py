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
# =============================================================================


from datetime import datetime
from pathlib import Path

import click
import numpy as np
import pandas as pd
import tensorflow as tf
from pydantic import PositiveInt

# from tensorflow.keras import mixed_precision
from rich.progress import Progress

import migration.datasets_original_ds as datasets_original_ds
from migration.config import DatasetConfig, OptimizedBound, TrainingConfig
from migration.datasets import TensorflowEncodedBatchedInferenceDatasetBuilder
from migration.models import vrnn
from migration.models.vrnn_elbo import VRNN
from migration.models.vrnn_fivo_predict import VRNNboundFIVO
from migration.utils import console

# mixed_precision.set_global_policy('mixed_float16')

def _get_config() -> TrainingConfig:
    return TrainingConfig(
    dataset=DatasetConfig(
        training_parquet='../new_data/ais_train_arrow.parquet',
        validation_parquet='../new_data/ais_val_arrow.parquet',
        test_parquet='../new_data/ais_test_arrow.parquet',
        mean_pickle='../data/ct_2017010203_10_20/mean.pkl',
        shuffle=False,
        val_size=15813 // 32,
        training_size=73795 // 32,
        test_size=494
    ), epochs=9999)



def get_pandas_generator(parquet : Path):
    def get_in_memory_dataset_generator():
        _ds = pd.read_parquet(parquet, columns=['latitude', 'longitude', 'sog', 'cog'])
        _idxs = _ds.index.unique()
        for idx in _idxs:
            track = _ds.loc[idx].values
            yield idx, track.reshape((-1, 4))
    return get_in_memory_dataset_generator



def build_test_dataset(cfg : DatasetConfig) -> tf.data.Dataset:
    test_generator = get_pandas_generator(cfg.test_parquet)
    test_dataset = TensorflowEncodedBatchedInferenceDatasetBuilder(
        track_with_id_generator=test_generator,
        batch_size=cfg.batch_size,
        lat_bins=cfg.encoding_bins.lat,
        lon_bins=cfg.encoding_bins.lon,
        sog_bins=cfg.encoding_bins.sog,
        cog_bins=cfg.encoding_bins.cog,
        shuffle=False,
        repeat=False
    ).build()
    return test_dataset

def create_model(mean_path : Path, latent_size : PositiveInt, total_bins : PositiveInt, bound : OptimizedBound, num_samples : PositiveInt):
    # Convert the mean of the training set to logit space so it can be used to
    # initialize the bias of the generative distribution.
    mean = datasets_original_ds.get_AIS_dataset_mean(mean_path)
    generative_bias_init = -tf.math.log(1. / tf.clip_by_value(mean, 0.0001, 0.9999) - 1)
    generative_distribution_class = vrnn.ConditionalBernoulliDistribution
    if bound == OptimizedBound.elbo:
        model = VRNN(total_bins,
                             latent_size,
                             generative_distribution_class,
                             generative_bias_init=generative_bias_init,
                             raw_sigma_bias=0.5, num_samples=1)
    else: 
        model = VRNNboundFIVO(
            total_bins,
                             latent_size,
                             generative_distribution_class,
                             generative_bias_init=generative_bias_init,
                             raw_sigma_bias=0.5, num_samples=num_samples
        )
    return model

def initialize_model(dataset : tf.data.Dataset, model : tf.keras.Model):
    _, inputs, targets, lengths = next(iter(dataset))
    model((inputs, targets), lengths)


def get_dataset_size(dataset : tf.data.Dataset):
    i = 0
    for _ in dataset:
        i += 1
    print(f'length {i}')
    return i

# def do_inference_step(batch : tuple[tf.Tensor], model : tf.keras.Model):

#     @tf.function(
#         input_signature=(
#             tf.TensorSpec(shape=[None,32,702], dtype=tf.float32,name='x'),
#             tf.TensorSpec(shape=[None,32,702], dtype=tf.float32, name='y'),
#             tf.TensorSpec(shape=[32], dtype=tf.int32, name='lengths'),
#         )
#     )
#     def inference_step(x,y, lengths, model):
#         return model((x, y),lengths)
#     inputs, targets, lengths = batch
#     return inference_step(inputs, targets, lengths)

class InferenceStep:
    def __init__(self, model):
        self.model = model 

    @tf.function(
        input_signature=(
            tf.TensorSpec(shape=[None,32,702], dtype=tf.float32,name='x'),
            tf.TensorSpec(shape=[None,32,702], dtype=tf.float32, name='y'),
            tf.TensorSpec(shape=[32], dtype=tf.int32, name='lengths'),
        )
    )
    def _do(self, x, y, lengths):
        return self.model((x, y),lengths)
    
    def do(self, batch : tuple[tf.Tensor]):
        _, inputs, targets, lengths = batch
        return self._do(inputs, targets, lengths)


@click.command()
@click.option("--exp", default=datetime.now().strftime("%Y_%m_%d-%H_%M_%S"), help="experiment name, defaults to timestamp %Y_%m_%d-%H_%M_%S")
@click.option("--models_dir", type=click.Path(path_type=Path, exists=False), default=Path("../output/models"), help="path to models saving")
def main(models_dir : Path, exp : str):
    cfg = _get_config()
    cfg.model.bound = OptimizedBound.fivo
    if cfg.random_seed:
        tf.random.set_seed(cfg.random_seed)

    # gather training data
    test_dataset = build_test_dataset(cfg.dataset)
    # test_dataset_length = get_dataset_size(test_dataset)
    test_dataset_length = cfg.dataset.test_size

    # dataset : tf.data.Dataset = create_dataset(cfg.dataset, repeat=False) 

   
    model = create_model(cfg.dataset.mean_pickle, cfg.model.latent_size, cfg.dataset.encoding_bins.total, cfg.model.bound, cfg.model.num_samples)
    # needed to initialize the LSTM weights
    initialize_model(test_dataset, model)

    # loading model
    models_dir = models_dir / exp
    assert models_dir.exists(), f'experiment with path {models_dir} does not exist' 
    model_path = models_dir / "vrnn.weights.h5"
    assert model_path.exists(), f'weights file of experiment {models_dir} does not exist' 
    model.load_weights(model_path)

    logprob_path = models_dir / 'logprob.pkl'
    fig_path = models_dir / 'logprob.png'
    logprob_ds = models_dir / 'logprob.parquet'

    l_dict = []

    logprob_mean = None

    steper = InferenceStep(model)


    @tf.function(
        input_signature=(
            tf.TensorSpec(shape=[None,16,32], dtype=tf.float32,name='log_weights'),
            tf.TensorSpec(shape=[32], dtype=tf.int32, name='lengths'),
        )
    )
    def calculate_batched_mean(log_weights,lengths):
        log_weights = tf.reshape(log_weights, (-1, cfg.dataset.batch_size))
        log_weights = tf.reduce_sum(log_weights, axis=0)
        return log_weights /(16 * tf.cast(lengths, tf.float32))


    # with Progress(console=console, transient=True) as progress:
    #     task = progress.add_task("Calculating logprob...", total=test_dataset_length)

    #     log_mean_arr = tf.TensorArray(tf.float32, test_dataset_length)

    #     for idx, batch in enumerate(test_dataset):
    #         progress.update(task, description=f"Batch {idx + 1}/{test_dataset_length}")

    #         _, _, lengths = batch
    #         # (max_len, particle_size, batch_size)
    #         log_weights = steper.do(batch)
    #         log_mean_arr = log_mean_arr.write(idx, calculate_batched_mean(log_weights, lengths))
    #         # unroll batch
    #         # for batch_idx in range(cfg.dataset.batch_size):
    #         #     seq_len_d = lengths[batch_idx]
    #         #     i_logprob_mean = np.mean(log_weights[:seq_len_d,:,batch_idx])
    #         #     logprob_mean = (logprob_mean+i_logprob_mean)/2
    #             # seq_len_d = lengths[batch_idx]
    #             # l_dict.append({
    #             #     'seq' : np.nonzero(targets[:seq_len_d,batch_idx,:])[1].reshape(-1,4),
    #             #     'log_weights_mean' : np.mean(log_weights[:seq_len_d,:,batch_idx]),
    #             # })
    #         progress.advance(task)
    # log_mean_arr = log_mean_arr.stack()



    # log_mean_arr = tf.reduce_mean(log_mean_arr)
    FIG_DPI = 150

    LOGPROB_MEAN_MIN = -10.0
    LOGPROB_STD_MAX = 5


    BATCH_SIZE = 32
    DATASET_TRACK_T_LEN = 989426
    
    output_data = {
        'track_id' : np.zeros(DATASET_TRACK_T_LEN, dtype=np.uint16),
        't' : np.zeros(DATASET_TRACK_T_LEN, dtype=np.uint16),
        'log_prob' : np.zeros(DATASET_TRACK_T_LEN, dtype=np.float32)
    }
    data_idx = 0

    with Progress(console=console, transient=True) as progress:
        task = progress.add_task("Calculating logprob...", total=test_dataset_length)

        # log_mean_arr = tf.TensorArray(tf.float32, test_dataset_length)
        # v_logprob = np.zeros(test_dataset_length*BATCH_SIZE)

        for idx, batch in enumerate(test_dataset):
            progress.update(task, description=f"Batch {idx + 1}/{test_dataset_length}")

            track_ids, _, _, lengths = batch
            # (max_len, particle_size, batch_size)
            batch_log_weights = steper.do(batch)
            # log_mean_arr = log_mean_arr.write(idx, calculate_batched_mean(log_weights, lengths))
            # unroll batch
            for track_idx in range(cfg.dataset.batch_size):
                track_len = lengths[track_idx]
                track_log_weights = batch_log_weights[:track_len,:,track_idx]
                # mean of all samples, tensor of shape t
                track_log_weights = np.mean(track_log_weights, axis=1)
                
                next_data_idx = data_idx + track_len
                
                output_data['track_id'][data_idx:next_data_idx] = track_ids[track_idx]
                output_data['t'][data_idx:next_data_idx] = range(track_len)
                output_data['log_prob'][data_idx:next_data_idx] = track_log_weights # already numpy
                
                data_idx = next_data_idx
            progress.advance(task)

    # cut unused
    output_data['track_id'] = output_data['track_id'][:data_idx]
    output_data['t'] = output_data['t'][:data_idx]
    output_data['log_prob'] = output_data['log_prob'][:data_idx]
                

    df = pd.DataFrame(output_data)
    df = df.set_index('track_id')
    df = df.set_index('t', append=True)
    df.to_parquet(logprob_ds)
    # d_mean = np.mean(v_logprob)
    # d_std = np.std(v_logprob)
    # d_thresh = d_mean - 3*d_std

    # plt.figure(figsize=(1920/FIG_DPI, 640/FIG_DPI), dpi=FIG_DPI)
    # plt.plot(v_logprob,'o')
    # plt.title(f"Log likelihood {cfg.dataset.test_parquet}, mean {d_mean:02f}, std {d_std:02f}, threshold {d_thresh:02f}")
    # plt.plot([0,len(v_logprob)], [d_thresh, d_thresh],'r')

    # plt.xlim([0,len(v_logprob)])
    # plt.savefig(fig_path,dpi = FIG_DPI)
    # plt.close()

if __name__ == "__main__":
    main()
