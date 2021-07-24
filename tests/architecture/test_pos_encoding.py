import pytest
import tensorflow as tf

from tests.tests_utils import tf_equal
from transformer.architecture.pos_encoding import PostionalEncoding

@pytest.fixture(scope="module")
def pe():
    return PostionalEncoding(seq_len=100, depth=512, max_denom=10000)


## PostionalEncoding
def test_init():
    pe = PostionalEncoding(seq_len=20, depth=100, max_denom=5000)
    assert pe.seq_len == 20, ("seq_len attribute does not match the "
                                  "expected value")
    assert pe.depth == 100, "depth attribute does not match the expected value"
    assert pe.max_denom == 5000, ("max_denom attribute does not match the "
                                  "expected value")


def test_alternate_concat(pe):
    # errors
    X = tf.random.normal((32, 10, 50))
    Y_wrong = tf.random.normal((31, 10, 50))
    with pytest.raises(ValueError, match=".*32.*31.*"):
        pe._alternate_concat(X, Y_wrong)
    # normal usage
    Y = tf.random.normal((32, 10, 50))
    concat = pe._alternate_concat(X, Y)
    assert concat.shape == tf.TensorShape((32, 10, 100)), (
        "the output of alternate_concat has not the expected shape"
        )
    assert tf_equal(X, concat[..., 0::2]) and tf_equal(Y, concat[..., 1::2]), (
        "the output of alternate_concat has not the expected values"
        )


def test_call(pe):
    # case depth even
    pos_code = pe()
    assert pos_code.shape == tf.TensorShape((100, 512)), (
        "the output has not the expected shape"
        )
    bounded_values = tf.logical_or(pos_code > -1.0, pos_code < 1.0)
    assert tf.reduce_all(bounded_values), "some values are outside [-1, 1]"
    # case depth odd
    pos_code = PostionalEncoding(seq_len=100, depth=511)()
    assert pos_code.shape == tf.TensorShape((100, 511)), (
        "the output has not the expected shape"
        )
