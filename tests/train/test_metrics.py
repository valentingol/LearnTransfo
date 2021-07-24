import math

import pytest
import tensorflow as tf

from tests.tests_utils import tf_equal
from transformer.train.metrics import MaskedSparseCategoricalCrossentropy
from transformer.train.metrics import MaskedAccuracy

@pytest.fixture(scope="module")
def y_true():
    return tf.constant([[0, 2, 1], [0, 1, 2]], dtype=tf.int64)

@pytest.fixture(scope="module")
def y_pred():
    return tf.constant([[[0.6, 0.2, 0.2],  # masked
                         [0.5, 0.3, 0.2],  # wrong
                         [0.2, 0.7, 0.1]], # good

                        [[0.8, 0.2, 0.0],  # masked
                         [0.3, 0.5, 0.2],  # good
                         [0.2, 0.7, 0.1]]] # wrong
                       )

@pytest.fixture(scope="module")
def mscc():
    return MaskedSparseCategoricalCrossentropy(from_logits=False)

@pytest.fixture(scope="module")
def maccuracy():
    return MaskedAccuracy()


# MaskedSparseCategoricalCrossentropy
def test_init_MSCC():
    mscc = MaskedSparseCategoricalCrossentropy(from_logits=False)
    SCC = tf.keras.losses.SparseCategoricalCrossentropy

    assert mscc.from_logits is False, ("from_logits attribute does not match "
                                       "the expected value")
    assert isinstance(mscc.base_loss, SCC), ("base_loss attribute is not an "
                                          "instance of the the expected class")


def test_call_MSCC(mscc, y_true, y_pred):
    loss = mscc(y_true, y_pred)
    scc_loss = tf.keras.losses.SparseCategoricalCrossentropy()
    assert loss.shape == (), "loss should be a scalar value (0-D tensor)"
    assert loss.dtype == tf.float32, "loss should be a float32 value"
    log = math.log
    expected_value = scc_loss(y_true[:, 1:], y_pred[:, 1:, :]).numpy()
    assert math.isclose(loss.numpy(), expected_value), ("loss should be close "
                                                        "to expected_value")


# MaskedAccuracy
def test_init_MAccuracy():
    pass


def test_call_MAccuracy(maccuracy, y_true, y_pred):
    acc = maccuracy(y_true, y_pred)
    assert acc.shape == (), "acc should be a scalar value (0-D tensor)"
    assert acc.dtype == tf.float32, "acc should be a float32 value"
    assert acc.numpy() == 0.5, "acc should be 0.5"
