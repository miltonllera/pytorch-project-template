import sys
from sacred.observers import FileStorageObserver

sys.path.insert(0, '../../src')

from experiment import create_experiment, Config


ex, run_experiment = create_experiment(
    task='deladd',
    name='Baseline subLSTM vs LSTM',
    dataset_configs=[
        Config(config='configs/dummy-dataset.yaml')],
    training_configs=[
        Config(config='configs/dummy-training.yaml')],
    model_configs=[
        Config(name='sublstm', config='configs/baseline-sublstm.yaml'),
        Config(name='lstm', config='configs/baseline-lstm.yaml')],
    observers=[
        FileStorageObserver('../../data/sims/')]
    )


@ex.automain
def main(_config, seed):
    run_experiment(_config, seed)
