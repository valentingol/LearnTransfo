import os
import sys

import pytest
import tensorflow as tf

myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../../')

from tests.tests_utils import tf_equal
from transformer.mask.mask import get_padding_mask, get_future_mask, get_masks

@pytest.fixture
def input_seq():
    return tf.constant([[6, 20, 87, 68, 14, 87, 11, 36, 0, 0],
                       [57, 18, 37, 9, 41, 47, 25, 31, 60, 0],
                       [82, 64, 3, 77, 69, 34, 76, 14, 46, 70],
                       [51, 21, 19, 61, 83, 0, 0, 0, 0, 0]])

@pytest.fixture
def output_seq():
    return tf.constant([[62, 83, 41, 77, 53, 2, 45, 16, 75, 19],
                        [73, 43, 5, 61, 37, 84, 52, 42, 0, 0],
                        [76, 19, 52, 43, 6, 22, 52, 0, 0, 0],
                        [85, 34, 15, 26, 74, 63, 8, 13, 43, 0],
                        [64, 84, 0, 0, 0, 0, 0, 0, 0, 0]])


def test_get_padding_mask(input_seq):
    padding_mask = get_padding_mask(input_seq)
    expected_mask = tf.constant([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                                 [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                                 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                 [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]])
    expected_mask = tf.reshape(expected_mask, (4, 1, 1, 10))
    assert padding_mask.shape == tf.TensorShape((4, 1, 1, 10)), (
        "padding_mask has not the expected shape"
        )
    assert tf_equal(padding_mask, expected_mask), ("padding mask do not match "
                                                   "the expected values")


def test_get_future_mask(output_seq):
    future_mask = get_future_mask(output_seq)
    expected_mask = tf.constant([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                                 [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                                 [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                                 [1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                                 [1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                                 [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                                 [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                                 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
    expected_mask = tf.reshape(expected_mask, (1, 1, 10, 10))
    assert future_mask.shape == tf.TensorShape(((1, 1, 10, 10))), (
        "padding_mask has not the expected shape"
        )
    assert tf_equal(future_mask, expected_mask), ("padding mask do not match "
                                                   "the expected values")

def test_get_masks(input_seq, output_seq):
    in_pad_mask, out_mask = get_masks(input_seq, output_seq)
    expected_in_pad_mask = tf.constant([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                                        [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                                        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                        [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]])
    expected_in_pad_mask = tf.reshape(expected_in_pad_mask, (4, 1, 1, 10))
    assert in_pad_mask.shape == tf.TensorShape(((4, 1, 1, 10))), (
        "padding_mask has not the expected shape"
        )
    assert tf_equal(in_pad_mask, expected_in_pad_mask), (
        "padding mask do not match the expected values"
        )
    expected_out_mask = tf.constant([
                              [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                               [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                               [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                               [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                               [1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                               [1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                               [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                               [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                               [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]],

                              [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                               [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                               [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                               [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                               [1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                               [1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                               [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                               [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                               [1, 1, 1, 1, 1, 1, 1, 1, 0, 0]],

                              [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                               [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                               [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                               [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                               [1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                               [1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                               [1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                               [1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                               [1, 1, 1, 1, 1, 1, 1, 0, 0, 0]],

                              [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                               [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                               [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                               [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                               [1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                               [1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                               [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                               [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                               [1, 1, 1, 1, 1, 1, 1, 1, 1, 0]],

                              [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                               [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                               [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                               [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                               [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                               [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                               [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                               [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                               [1, 1, 0, 0, 0, 0, 0, 0, 0, 0]]
                                ])
    expected_out_mask = tf.reshape(expected_out_mask, (5, 1, 10, 10))
    assert out_mask.shape == tf.TensorShape((5, 1, 10, 10)), (
        "padding_mask has not the expected shape"
        )
    assert tf_equal(out_mask, expected_out_mask), (
        "padding mask do not match the expected values"
        )
