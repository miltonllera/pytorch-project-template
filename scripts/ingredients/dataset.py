import sys
from sacred import Ingredient

if sys.path[0] != '../src':
    sys.path.insert(0, '../src')

from dataset.deladd import load_delayed_addition
from dataset.mnist import load_mnist

deladd = Ingredient('dataset')
load_deladd = deladd.capture(load_delayed_addition)

seqmnist = Ingredient('dataset')
load_seqmnist = seqmnist.capture(load_mnist)
