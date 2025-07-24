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
from rich.progress import Progress

# from tensorflow.keras import mixed_precision
from tqdm import tqdm

import migration.datasets_original_ds as datasets_original_ds
from migration.config import DatasetConfig, OptimizedBound, TrainingConfig
from migration.datasets import TensorflowEncodedBatchedDatasetBuilder
from migration.models import vrnn
from migration.models.vrnn_elbo import VRNN
from migration.models.vrnn_fivo import VRNNboundFIVO
from migration.utils import AverageMeter, console, logger

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



def get_datasets(cfg : DatasetConfig) -> tuple[tf.data.Dataset, tf.data.Dataset]:
    train_generator = get_pandas_generator(cfg.training_parquet)
    val_generator = get_pandas_generator(cfg.validation_parquet)
    train = TensorflowEncodedBatchedDatasetBuilder(
        track_generator=train_generator,
        batch_size=cfg.batch_size,
        lat_bins=cfg.encoding_bins.lat,
        lon_bins=cfg.encoding_bins.lon,
        sog_bins=cfg.encoding_bins.sog,
        cog_bins=cfg.encoding_bins.cog,
        shuffle=cfg.shuffle,
        repeat=True
    ).build()

    val = TensorflowEncodedBatchedDatasetBuilder(
        track_generator=val_generator,
        batch_size=cfg.batch_size,
        lat_bins=cfg.encoding_bins.lat,
        lon_bins=cfg.encoding_bins.lon,
        sog_bins=cfg.encoding_bins.sog,
        cog_bins=cfg.encoding_bins.cog,
        shuffle=cfg.shuffle,
        repeat=True
    ).build()
            
    return train, val
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

def initialize_model(train_dataset : tf.data.Dataset, model : tf.keras.Model):
    inputs, targets, lengths = next(iter(train_dataset))
    model((inputs, targets), lengths)

def get_dataset_size(dataset : tf.data.Dataset):
    return sum(1 for _ in dataset)

class TrainEpochManager:
    def __init__(self, model, optimizer, dataset, dataset_size : int | None = None):
        self.model = model 
        self.optimizer = optimizer
        self.dataset = dataset
        if dataset_size is None:
            logger.info('training dataset size not provided, calculating...')
            self.dataset_size = get_dataset_size(dataset)

    @tf.function(
        input_signature=(
            tf.TensorSpec(shape=[None,32,702], dtype=tf.float32,name='x'),
            tf.TensorSpec(shape=[None,32,702], dtype=tf.float32, name='y'),
            tf.TensorSpec(shape=[32], dtype=tf.int32, name='lengths'),
        )
    )
    def _step(self, x, y, lengths):
        with tf.GradientTape() as tape:
            log_likelihood = self.model((x, y),lengths)
            # Compute lower bounds on the log likelihood.
            log_likelihood = tf.reduce_mean(input_tensor=log_likelihood / tf.cast(lengths, dtype=tf.float32))
            loss = -log_likelihood
        grads = tape.gradient(loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        return loss

    def do_epoch(self):
        losses = AverageMeter()
        with Progress(console=console, transient=True) as progress:
            task = progress.add_task("Training...", total=self.dataset_size)
            # ugly zip since dataset iterator continues fetching by default
            for idx, batch in zip(range(self.dataset_size),self.dataset):
                progress.update(task, description=f"Batch {idx + 1}/{self.dataset_size}")
                inputs, targets, lengths = batch
                loss = self._step(inputs, targets, lengths)
                losses.update(loss)
                progress.advance(task)
        return losses.avg


class ValidationEpochManager:
    def __init__(self, model, dataset, dataset_size : int | None = None):
        self.model = model
        self.dataset = dataset
        self.dataset_size = get_dataset_size(dataset)
        if dataset_size is None:
            logger.info('validation dataset size not provided, calculating...')
            self.dataset_size = get_dataset_size(dataset)

    @tf.function(
        input_signature=(
            tf.TensorSpec(shape=[None,32,702], dtype=tf.float32,name='x'),
            tf.TensorSpec(shape=[None,32,702], dtype=tf.float32, name='y'),
            tf.TensorSpec(shape=[32], dtype=tf.int32, name='lengths'),
        )
    )
    def _step(self, x, y, lengths):
        log_likelihood = self.model((x, y),lengths)
        # Compute lower bounds on the log likelihood.
        log_likelihood = tf.reduce_mean(input_tensor=log_likelihood / tf.cast(lengths, dtype=tf.float32))
        return log_likelihood


    def do_epoch(self):
        log_likelihood = AverageMeter()
        for _, batch in zip(range(self.dataset_size),self.dataset):
            inputs, targets, lengths = batch
            loss = self._step(inputs, targets, lengths)
            log_likelihood.update(loss)
        return log_likelihood.avg



# def do_train_epoch(train_dataset : tf.data.Dataset, model : tf.keras.Model, optimizer :tf.keras.Optimizer, train_size : PositiveInt):
#     losses = AverageMeter()

#     @tf.function(
#         input_signature=(
#             tf.TensorSpec(shape=[None,32,702], dtype=tf.float32,name='x'),
#             tf.TensorSpec(shape=[None,32,702], dtype=tf.float32, name='y'),
#             tf.TensorSpec(shape=[32], dtype=tf.int32, name='lengths'),
#         )
#     )
#     def train_step(x,y, lengths):
#         with tf.GradientTape() as tape:
#             log_likelihood = model((x, y),lengths)
#             # Compute lower bounds on the log likelihood.
#             log_likelihood = tf.reduce_mean(input_tensor=log_likelihood / tf.cast(lengths, dtype=tf.float32))
#             loss = -log_likelihood
#         grads = tape.gradient(loss, model.trainable_weights)
#         optimizer.apply_gradients(zip(grads, model.trainable_weights))
#         return loss
    
#     with Progress(console=console, transient=True) as progress:
#         task = progress.add_task("Training...", total=train_size)
#         # ugly zip since dataset iterator continues fetching by default
#         for idx, batch in zip(range(train_size),train_dataset):
#             progress.update(task, description=f"Batch {idx + 1}/{train_size}")
#             inputs, targets, lengths = batch
#             loss = train_step(inputs, targets, lengths)
#             losses.update(loss)
#             progress.advance(task)
#     return losses.avg


# def do_validate_epoch(val_dataset : tf.data.Dataset, model : tf.keras.Model, val_size : PositiveInt):
#     log_likelihood = AverageMeter()

#     @tf.function(
#         input_signature=(
#             tf.TensorSpec(shape=[None,32,702], dtype=tf.float32,name='x'),
#             tf.TensorSpec(shape=[None,32,702], dtype=tf.float32, name='y'),
#             tf.TensorSpec(shape=[32], dtype=tf.int32, name='lengths'),
#         )
#     )
#     def validate_step(x,y, lengths):
#         log_likelihood = model((x, y),lengths)
#         # Compute lower bounds on the log likelihood.
#         log_likelihood = tf.reduce_mean(input_tensor=log_likelihood / tf.cast(lengths, dtype=tf.float32))
#         return log_likelihood
    
#     # ugly zip since dataset iterator continues fetching by default
#     for idx, batch in zip(range(val_size-1),val_dataset):
#         inputs, targets, lengths = batch
#         loss = validate_step(inputs, targets, lengths)
#         log_likelihood.update(loss)
#     return log_likelihood.avg

    # ugly zip since dataset iterator continues fetching by default

# def run_train(cfg : TrainingConfig):

#     if cfg.random_seed:
#         tf.random.set_seed(cfg.random_seed)

#     train_dataset, val_dataset = get_cached_datasets()
#     # dataset : tf.data.Dataset = create_dataset(cfg.dataset, repeat=False)    
#     model = create_model(cfg.dataset.mean_pickle, cfg.model.latent_size, cfg.dataset.encoding_bins.total)
#     optimizer = tf.keras.optimizers.Adam(learning_rate=cfg.learning_rate)
    
#     @tf.function(reduce_retracing=True)
#     def train_step(x,y, lengths):
#         with tf.GradientTape() as tape:
#             bound = model((x, y),lengths)
#             # Compute lower bounds on the log likelihood.
#             bound = tf.reduce_mean(input_tensor=bound / tf.cast(lengths, dtype=tf.float32))
#             loss = -bound
#         grads = tape.gradient(loss, model.trainable_weights)
#         optimizer.apply_gradients(zip(grads, model.trainable_weights))
#         return loss
    
#     ckpt = './here.weights.h5'

#     if Path(ckpt).exists():
#         inputs, targets, lengths = next(iter(dataset))
#         _ = model((inputs, targets), lengths)
#         print('loading previous weights')
#         model.load_weights(ckpt)
    
#     for _ in tqdm(range(cfg.epochs)):
#         for batch, (inputs, targets, lengths) in enumerate(dataset):
#             loss = train_step(inputs, targets, lengths)
#             if batch % 50 == 0:
#                 print(f'batch {batch}, loss {loss}')
#         model.save_weights(ckpt, overwrite=True)

@click.command()
@click.option("--exp", default=datetime.now().strftime("%Y_%m_%d-%H_%M_%S"), help="experiment name, defaults to timestamp %Y_%m_%d-%H_%M_%S")
@click.option("--models_dir", type=click.Path(path_type=Path, exists=False), default=Path("../output/models"), help="path to models saving")
def main(models_dir : Path, exp : str):
    cfg = _get_config()
    cfg.model.bound = OptimizedBound.fivo
    if cfg.random_seed:
        tf.random.set_seed(cfg.random_seed)

    # gather training data
    train_dataset, val_dataset = get_datasets(cfg.dataset)
    # dataset : tf.data.Dataset = create_dataset(cfg.dataset, repeat=False) 

   
    model = create_model(cfg.dataset.mean_pickle, cfg.model.latent_size, cfg.dataset.encoding_bins.total, cfg.model.bound, cfg.model.num_samples)
    # needed to initialize the LSTM weights
    initialize_model(train_dataset, model)
    # optimizer = tf.keras.optimizers.Adam(learning_rate=cfg.learning_rate)
    optimizer = tf.keras.optimizers.AdamW(learning_rate=cfg.learning_rate)

    
    # checkpointing logic
    models_dir = models_dir / exp
    models_dir.mkdir(exist_ok=True, parents=True)
    ckpt = tf.train.Checkpoint(model = model, 
                                     optimizer = optimizer, 
                                     epoch = tf.Variable(0),
                                     best_log_likelihood = tf.Variable(-np.inf, dtype=float))
    
    chpt_mgr = tf.train.CheckpointManager(ckpt, models_dir / 'checkpoints', max_to_keep=3)

    if chpt_mgr.latest_checkpoint:
        logger.info(f"Checkpoint exists, resuming from {chpt_mgr.latest_checkpoint}")
        ckpt.restore(chpt_mgr.latest_checkpoint)
    else:
        logger.info("Checkpoint does not exist, starting from scratch")

    model_path = models_dir / "vrnn.weights.h5"
    with (models_dir / 'metadata.txt').open(mode='w') as f:
        f.write(f'type {cfg.model.bound}\nlatent size {cfg.model.latent_size}')

    with (models_dir / 'best_log_likelihood.txt').open(mode='w') as f:
        f.write(f'{ckpt.best_log_likelihood.numpy():.2f}')
    # Training loop
    start_epoch = int(ckpt.epoch)
    train_epoch_manager = TrainEpochManager(model, optimizer, train_dataset, cfg.dataset.training_size)
    val_epoch_manager = ValidationEpochManager(model, val_dataset, cfg.dataset.val_size)

    for epoch in tqdm(range(start_epoch, cfg.epochs)):
        logger.info(f"Training epoch {epoch}...")
        loss = train_epoch_manager.do_epoch()
        logger.info(f"Training epoch {epoch}\tLoss: {loss:.2f}")

        if epoch % cfg.eval_frequency == 0:
            logger.info(f"Validating epoch {epoch}...")
            avg_log_likelihood = val_epoch_manager.do_epoch()
            logger.info(f"Validation epoch {epoch}\tLog Likelihood: {avg_log_likelihood:.2f}\tLikelihood: {np.exp(avg_log_likelihood):.2e}")
            if avg_log_likelihood > ckpt.best_log_likelihood:
                logger.info(f"Log likelihood improved {ckpt.best_log_likelihood.numpy():.2f} -> {avg_log_likelihood.numpy():.2f}, saving model...")
                logger.info(f"  Likelihood improvement: {np.exp(ckpt.best_log_likelihood.numpy()):.2e} -> {np.exp(avg_log_likelihood.numpy()):.2e}")
                ckpt.best_log_likelihood.assign(avg_log_likelihood)
                model.save_weights(model_path, overwrite=True)
                with (models_dir / 'best_log_likelihood.txt').open(mode='w') as f:
                    f.write(f'{avg_log_likelihood:.2f}')
        ckpt.epoch.assign_add(1)
        # checkpoint
        chpt_mgr.save()
        # quickview
        with (models_dir / 'epochs.txt').open(mode='w') as f:
            f.write(f'{epoch}')
        


if __name__ == "__main__":
    main()
