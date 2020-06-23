import sys
from sacred import Ingredient

if sys.path[0] != '../src':
    sys.path.insert(0, '../src')

from model.rnn import init_rnn


model = Ingredient('model')
init_rnn = model.capture(init_rnn)


@model.capture
def init_model(device):
    model = init_rnn()
    return model.to(device=device)


@model.command(unobserved=True)
def show():
    model = init_model()
    print(model)