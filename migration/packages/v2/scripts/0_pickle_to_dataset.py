from pathlib import Path

import migration.datasets as datasets
from migration.config import DatasetConfig


def _export_pickles_as_datasets(cfg : DatasetConfig):
    datadir = Path('../data/tensorflow')
    for path, export_name in zip([cfg.training_pickle, cfg.validation_pickle, cfg.test_pickle], ['train', 'validation', 'test'],):
        dataset = datasets.get_Tensorflow_AIS_dataset(
                    str(path),
                    cfg.batch_size,
                    cfg.encoding_bins.lat,
                    cfg.encoding_bins.lon, 
                    cfg.encoding_bins.sog,
                    cfg.encoding_bins.cog, 
                    shuffle=cfg.shuffle, 
                    repeat=False)
        dataset.save(str(datadir / export_name))
    
def main():
    cfg = DatasetConfig(
        training_pickle='../data/ct_2017010203_10_20/ct_2017010203_10_20_train.pkl',
        validation_pickle='../data/ct_2017010203_10_20/ct_2017010203_10_20_valid.pkl',
        test_pickle='../data/ct_2017010203_10_20/ct_2017010203_10_20_test.pkl',
        mean_pickle='../data/ct_2017010203_10_20/mean.pkl',
        shuffle=True
    )
    _export_pickles_as_datasets(cfg)

if __name__ == "__main__":
    main()
