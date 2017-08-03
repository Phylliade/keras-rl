from collections import namedtuple
import tensorflow as tf

RLVariables = namedtuple("Variables", ["state", "action"])


class Variable(tf.Tensor):
    pass
