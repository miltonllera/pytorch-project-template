import sys
from sacred import Ingredient
from ignite.engine import Events


if sys.path[0] != '../src':
    sys.path.insert(0, '../src')

from training.handlers import *
from training.loss import init_metrics
from training.optimizer import init_optimizer
from training.engine import create_rnn_trainer, create_rnn_evaluator

training = Ingredient('training')

init_metrics = training.capture(init_metrics)
init_optimizer = training.capture(training_optimizer)
create_rnn_trainer = training.capture(create_rnn_trainer)
create_rnn_evaluator = training.capture(create_rnn_evalautor)
