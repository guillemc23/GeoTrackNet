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

import migration.datasets_original_ds as datasets_original_ds
from migration.config import DatasetConfig, OptimizedBound, TrainingConfig
from migration.datasets import TensorflowEncodedBatchedDatasetBuilder
from migration.models import vrnn
from migration.models.vrnn_elbo import VRNN
from migration.models.vrnn_fivo import VRNNboundFIVO
from migration.utils import AverageMeter, logger

# from tensorflow.keras import mixed_precision


# mixed_precision.set_global_policy('mixed_float16')

def _get_config() -> TrainingConfig:
    return TrainingConfig(
    dataset=DatasetConfig(
        training_parquet='../new_data/ais_train.parquet',
        validation_parquet='../new_data/ais_val.parquet',
        test_parquet='../new_data/ais_test.parquet',
        mean_pickle='../data/ct_2017010203_10_20/mean.pkl',
        shuffle=False,
        val_size=15813 // 32,
        training_size=73795 // 32,
    ), epochs=9999)


def get_pandas_generator(parquet : Path):
    def get_in_memory_dataset_generator():
        _ds = pd.read_parquet(parquet).replace(1, 0.9999)
        _ds.insert(0, 'temp_id', range(0, len(_ds)))
        _ds = _ds.set_index('temp_id', append=True)
        _ds = _ds.sort_index().reset_index(1).drop('temp_id', axis=1)
        _idxs = _ds.index.unique()
        for idx in range(len(_idxs)):
            track = _ds.loc[_idxs[idx]].values
            yield track.reshape((-1, 4)).astype(np.float32)
    return get_in_memory_dataset_generator



def build_test_dataset(cfg : DatasetConfig) -> tf.data.Dataset:
    test_generator = get_pandas_generator(cfg.test_parquet)
    test_dataset = TensorflowEncodedBatchedDatasetBuilder(
        track_generator=test_generator,
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
    inputs, targets, lengths = next(iter(dataset))
    model((inputs, targets), lengths)


def do_test_epoch(test_dataset : tf.data.Dataset, model : tf.keras.Model):
    log_likelihood = AverageMeter()

    @tf.function(
        input_signature=(
            tf.TensorSpec(shape=[None,32,702], dtype=tf.float32,name='x'),
            tf.TensorSpec(shape=[None,32,702], dtype=tf.float32, name='y'),
            tf.TensorSpec(shape=[32], dtype=tf.int32, name='lengths'),
        )
    )
    def test_step(x,y, lengths):
        log_likelihood = model((x, y),lengths)
        # Compute lower bounds on the log likelihood.
        log_likelihood = tf.reduce_mean(input_tensor=log_likelihood / tf.cast(lengths, dtype=tf.float32))
        return log_likelihood
    
    # ugly zip since dataset iterator continues fetching by default
    for batch in test_dataset:
        inputs, targets, lengths = batch
        loss = test_step(inputs, targets, lengths)
        log_likelihood.update(loss)
    return log_likelihood.avg


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

    avg_log_likelihood = do_test_epoch(test_dataset, model)
    logger.info(f"Test dataset \tLog Likelihood: {avg_log_likelihood:.2f}\tLikelihood: {np.exp(avg_log_likelihood):.2e}")

if __name__ == "__main__":
    main()
