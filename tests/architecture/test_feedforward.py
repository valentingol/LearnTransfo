import os
import sys

import pytest
import tensorflow as tf

myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../../')

from transformer.architecture.feedforward import FeedForward


@pytest.fixture
def ff():
    return FeedForward(d_model=512, d_ff=1024, activation='relu')


## FeedForward
def test_init_feedforward():
    ff = FeedForward(d_model=512, d_ff=1024, activation='relu')
    assert ff.d_model == 512, ("d_model attribute does not match the "
                               "expected value")
    assert ff.d_ff == 1024, ("d_ff attribute does not match the "
                             "expected value")
    assert ff.activation == 'relu', ("activation attribute does not match the "
                                     "expected value")
    # test with callable activation
    def activation(X: tf.Tensor):
        return tf.maximum(X, 0.0)
    FeedForward(d_model=512, d_ff=1024, activation=activation)


def test_call_feedforward(ff):
    X = tf.random.normal((32, 100, 512))
    output = ff(X)
    assert output.shape == X.shape, ("the output shape do not match the input "
                                     "shape")
