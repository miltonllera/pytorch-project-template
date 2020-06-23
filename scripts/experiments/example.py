import sys
import torch
import ignite
from collections import namedtuple
from ignite.engine import Events
from sacred import Experiment
from sacred.observers import FileStorageObserver, MongoObserver

# Load experiment ingredients
from ingredients.dataset import deladd, load_deladd as load_dataset
from ingredients.model import model, init_model
from ingredients.training import training, init_metrics, init_optimizer, \
                                 create_rnn_trainer, create_rnn_evaluator, \
                                 Tracer, ModelCheckpoint, LRScheduler


# Add configs
training.add_config('configs/dummy-training.yaml')
deladd.add_config('configs/dummy-dataset.yaml')
model.add_config('dummy-lstm.yaml')


# Set up experiment
ex = Experiment(name='deladd', ingredients=[deladd, model, training])
ex.add_config(no_cuda=False, save_folder = '../../data/sims/deladd/temp/')
ex.add_package_dependency('torch', torch.__version__)
ex.observers.append(FileStorageObserver.create('../data/sims/test/'))


# Functions
@ex.capture
def set_seed_and_device(seed, no_cuda):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available() and not no_cuda:
        torch.cuda.manual_seed(seed)
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    return device


@ex.automain
def main(_config, seed):
    no_cuda = _config['no_cuda']
    batch_size = _config['training']['batch_size']
    save_folder = _config['save_folder']

    device = set_seed_and_device(seed, no_cuda)
    training_set, test_set, validation_set = load_dataset(batch_size=batch_size)

    model = init_model(device=device)

    # Init metrics
    loss, metrics = init_metrics('mse', ['mse'])
    optimizer = init_optimizer(params=model.parameters())

    # Init engines
    trainer = create_rnn_trainer(model, optimizer, loss, device=device)
    validator = create_rnn_evaluator(model, metrics, device=device)

    @trainer.on(Events.EPOCH_COMPLETED)
    def validate(engine):
        validator.run(validation_set)

    # Exception for early termination
    @trainer.on(Events.EXCEPTION_RAISED)
    def terminate(engine, exception):
        if isinstance(exception, KeyboardInterrupt):
            engine.should_terminate=True

    # Record training progression
    tracer = Tracer(metrics).attach(trainer)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training(engine):
        ex.log_scalar('training_loss', tracer.loss[-1])
        tracer.loss.clear()

    @validator.on(Events.EPOCH_COMPLETED)
    def log_validation(engine):
        for metric, value in engine.state.metrics.items():
            ex.log_scalar('val_{}'.format(metric), value)

    # Attach model checkpoint
    def score_fn(engine):
        return -engine.state.metrics[list(metrics)[0]]

    checkpoint = ModelCheckpoint(
        dirname=_config['save_folder'],
        filename_prefix='disent',
        score_function=score_fn,
        create_dir=True,
        require_empty=False,
        save_as_state_dict=True
    )
    validator.add_event_handler(Events.COMPLETED, checkpoint, {'model': model})

    # Run the training
    trainer.run(training_set, max_epochs=epochs)
    # Select best model
    model.load_state_dict(checkpoint.best_model)

    # Run on test data
    tester = create_supervised_evaluator(model, metrics, device=device)
    test_metrics = tester.run(test_set).metrics

    # Save best model performance and state
    for metric, value in test_metrics.items():
        ex.log_scalar('test_{}'.format(metric), value)

    ex.add_artifact(checkpoint.best_model_path, 'trained-model')