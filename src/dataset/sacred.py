from sacred import Ingredient
from .deladd import load_delayed_addition
from .mnist import load_mnist

deladd_data = Ingredient('dataset')
load_delayed_addition = deladd_data.capture(load_delayed_addition)

mnist_data = Ingredient('dataset')
load_mnist = mnist_data.capture(load_mnist)


def get_dataset_ingredient(task):
    if  task == 'deladd':
        return deladd_data, load_delayed_addition
    elif task == 'seqmnist':
        return mnist_data, load_mnist
    raise ValueError('Unrecognised task {}'.format(task))
