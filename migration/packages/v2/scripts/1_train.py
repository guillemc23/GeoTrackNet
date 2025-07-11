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
import tensorflow as tf
from pydantic import PositiveInt
from rich.progress import Progress

# from tensorflow.keras import mixed_precision
from tqdm import tqdm

import migration.datasets as datasets
from migration.config import DatasetConfig, OptimizedBound, TrainingConfig
from migration.models import vrnn
from migration.models.vrnn_elbo import VRNN
from migration.models.vrnn_fivo import VRNNboundFIVO
from migration.utils import AverageMeter, console, logger

# mixed_precision.set_global_policy('mixed_float16')

def _get_config() -> TrainingConfig:
    return TrainingConfig(
    dataset=DatasetConfig(
        training_pickle='../data/ct_2017010203_10_20/ct_2017010203_10_20_train.pkl',
        validation_pickle='../data/ct_2017010203_10_20/ct_2017010203_10_20_valid.pkl',
        test_pickle='../data/ct_2017010203_10_20/ct_2017010203_10_20_test.pkl',
        mean_pickle='../data/ct_2017010203_10_20/mean.pkl',
        shuffle=True
    ), epochs=9999)


def get_cached_datasets() -> tuple[tf.data.Dataset, tf.data.Dataset]:
    train = tf.data.Dataset.load('../data/tensorflow/train')
    validation = tf.data.Dataset.load('../data/tensorflow/validation')
    return train, validation


def get_datasets(cfg : DatasetConfig) -> tuple[tf.data.Dataset, tf.data.Dataset]:
    train = datasets.get_Tensorflow_AIS_dataset(
                    cfg.training_pickle,
                    cfg.batch_size,
                    cfg.encoding_bins.lat,
                    cfg.encoding_bins.lon, 
                    cfg.encoding_bins.sog,
                    cfg.encoding_bins.cog, 
                    shuffle=cfg.shuffle,
                    repeat=True)
    validation = datasets.get_Tensorflow_AIS_dataset(
                    cfg.validation_pickle,
                    cfg.batch_size,
                    cfg.encoding_bins.lat,
                    cfg.encoding_bins.lon, 
                    cfg.encoding_bins.sog,
                    cfg.encoding_bins.cog, 
                    shuffle=cfg.shuffle,
                    repeat=True)
    return train, validation

# get batch and model
def create_dataset(cfg: DatasetConfig) -> tf.data.Dataset:

    return datasets.get_Tensorflow_AIS_dataset(
                    cfg.training_pickle,
                    cfg.batch_size,
                    cfg.encoding_bins.lat,
                    cfg.encoding_bins.lon, 
                    cfg.encoding_bins.sog,
                    cfg.encoding_bins.cog, 
                    shuffle=cfg.shuffle,
                    repeat=False)

def create_model(mean_path : Path, latent_size : PositiveInt, total_bins : PositiveInt, bound : OptimizedBound, num_samples : PositiveInt):
    # Convert the mean of the training set to logit space so it can be used to
    # initialize the bias of the generative distribution.
    mean = datasets.get_AIS_dataset_mean(mean_path)
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



def do_train_epoch(train_dataset : tf.data.Dataset, model : tf.keras.Model, optimizer :tf.keras.Optimizer, train_size : PositiveInt):
    losses = AverageMeter()

    @tf.function(
        input_signature=(
            tf.TensorSpec(shape=[None,32,702], dtype=tf.float32,name='x'),
            tf.TensorSpec(shape=[None,32,702], dtype=tf.float32, name='y'),
            tf.TensorSpec(shape=[32], dtype=tf.int32, name='lengths'),
        )
    )
    def train_step(x,y, lengths):
        with tf.GradientTape() as tape:
            log_likelihood = model((x, y),lengths)
            # Compute lower bounds on the log likelihood.
            log_likelihood = tf.reduce_mean(input_tensor=log_likelihood / tf.cast(lengths, dtype=tf.float32))
            loss = -log_likelihood
        grads = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        return loss
    
    with Progress(console=console, transient=True) as progress:
        task = progress.add_task("Training...", total=train_size)
        # ugly zip since dataset iterator continues fetching by default
        for idx, batch in zip(range(train_size),train_dataset):
            progress.update(task, description=f"Batch {idx + 1}/{train_size}")
            inputs, targets, lengths = batch
            loss = train_step(inputs, targets, lengths)
            losses.update(loss)
            progress.advance(task)
    return losses.avg


def do_validate_epoch(val_dataset : tf.data.Dataset, model : tf.keras.Model, val_size : PositiveInt):
    log_likelihood = AverageMeter()

    @tf.function(
        input_signature=(
            tf.TensorSpec(shape=[None,32,702], dtype=tf.float32,name='x'),
            tf.TensorSpec(shape=[None,32,702], dtype=tf.float32, name='y'),
            tf.TensorSpec(shape=[32], dtype=tf.int32, name='lengths'),
        )
    )
    def validate_step(x,y, lengths):
        log_likelihood = model((x, y),lengths)
        # Compute lower bounds on the log likelihood.
        log_likelihood = tf.reduce_mean(input_tensor=log_likelihood / tf.cast(lengths, dtype=tf.float32))
        return log_likelihood
    
    # ugly zip since dataset iterator continues fetching by default
    for idx, batch in zip(range(val_size-1),val_dataset):
        inputs, targets, lengths = batch
        loss = validate_step(inputs, targets, lengths)
        log_likelihood.update(loss)
    return log_likelihood.avg

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

    # Training loop
    start_epoch = int(ckpt.epoch)

    for epoch in tqdm(range(start_epoch, cfg.epochs)):
        logger.info(f"Training epoch {epoch}...")
        loss = do_train_epoch(train_dataset, model, optimizer, cfg.dataset.training_size)
        logger.info(f"Training epoch {epoch}\tLoss: {loss:.2f}")

        if epoch % cfg.eval_frequency == 0:
            logger.info(f"Validating epoch {epoch}...")
            avg_log_likelihood = do_validate_epoch(val_dataset, model, cfg.dataset.val_size)
            logger.info(f"Validation epoch {epoch}\tLog Likelihood: {avg_log_likelihood:.2f}\tLikelihood: {np.exp(avg_log_likelihood):.2e}")
            if avg_log_likelihood > ckpt.best_log_likelihood:
                logger.info(f"Log likelihood improved {ckpt.best_log_likelihood.numpy():.2f} -> {avg_log_likelihood.numpy():.2f}, saving model...")
                logger.info(f"  Likelihood improvement: {np.exp(ckpt.best_log_likelihood.numpy()):.2e} -> {np.exp(avg_log_likelihood.numpy()):.2e}")
                ckpt.best_log_likelihood.assign(avg_log_likelihood)
                model.save_weights(model_path, overwrite=True)
        
        ckpt.epoch.assign_add(1)
        # checkpoint
        chpt_mgr.save()


if __name__ == "__main__":
    main()
