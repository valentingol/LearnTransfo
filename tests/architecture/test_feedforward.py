import pytest
import tensorflow as tf

from transformer.architecture.feedforward import FeedForward


@pytest.fixture(scope="module")
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

    ff = FeedForward(d_model=512, d_ff=1024, activation='gelu')
    assert ff.activation == 'gelu', ("activation attribute does not match the "
                                     "expected value")
    # test with callable activation
    activation = lambda X: tf.maximum(X, 0.0)
    FeedForward(d_model=512, d_ff=1024, activation=activation)


def test_call_feedforward(ff):
    X = tf.random.normal((32, 100, 512))
    output = ff(X)
    assert output.shape == X.shape, ("the output shape do not match the input "
                                     "shape")
