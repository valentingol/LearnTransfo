import pytest
import tensorflow as tf

from tests.tests_utils import tf_equal
from transformer.architecture.attention import MultiHeadAttention

@pytest.fixture(scope="module")
def mha():
    return MultiHeadAttention(d_model=512, n_heads=8)


## MultiHeadAttention class
def test_init():
    kl = tf.keras.layers
    # errors
    with pytest.raises(ValueError):
        MultiHeadAttention(d_model=512, n_heads=7)
    # normal usage
    mha = MultiHeadAttention(d_model=512, n_heads=8)
    assert mha.d_model == 512, ("d_model attribute does not match the "
                                "expected value")
    assert mha.n_heads == 8, ("n_heads attribute does not match the "
                              "expected value")
    assert mha.d_head == 64, ("d_head attribute does not match the expected "
                              "value")
    assert isinstance(mha.Wq, kl.Layer), ("Wq attribute has not the "
                                              "expected type")
    assert isinstance(mha.Wk, kl.Layer), ("Wk attribute has not the "
                                              "expected type")
    assert isinstance(mha.Wv, kl.Layer), ("Wv attribute has not the "
                                              "expected type")
    assert isinstance(mha.Wo, kl.Layer), ("Wo attribute has not the "
                                              "expected type")


def test_split_to_heads(mha):
    # errors
    X = tf.random.normal((32, 200, 511))
    with pytest.raises(ValueError, match=".*512.*"):
        mha._split_to_heads(X)
    # normal usage
    X = tf.random.normal((32, 200, 512))
    X_split = mha._split_to_heads(X)
    assert X_split.shape == tf.TensorShape((32, 8, 200, 64)), (
        "output of split_to_heads has not the expected shape"
        )


def test_scaled_dot_product_attention(mha):
    Q = tf.random.normal((32, 8, 100, 64))
    K = tf.random.normal((32, 8, 200, 64))
    V = tf.random.normal((32, 8, 200, 128))
    #errors
    K_wrong = tf.random.normal((32, 8, 200, 63))
    with pytest.raises(ValueError, match=".*64.*"):
        mha.scaled_dot_product_attention(Q, K_wrong, V)
    V_wrong = tf.random.normal((32, 8, 199, 128))
    with pytest.raises(ValueError, match=".*200.*"):
        mha.scaled_dot_product_attention(Q, K, V_wrong)
    # normal usage
    out1, att1 = mha.scaled_dot_product_attention(Q, K, V)
    assert out1.shape == tf.TensorShape((32, 8, 100, 128)), (
        "output of scaled_dot_product_attention has not the expected shape"
        )
    assert att1.shape == tf.TensorShape((32, 8, 100, 200)), (
        "attention of scaled_dot_product_attention has not the expected shape"
        )
    # mask
    mask1 = tf.ones((32, 1, 1, 199), dtype=tf.bool)
    mask0 = tf.zeros((32, 1, 1, 1), dtype=tf.bool)
    mask = tf.concat([mask0, mask1], axis=-1)
    _, att2 = mha.scaled_dot_product_attention(Q, K, V, mask)
    assert tf_equal(att2[:, 0, 0, 0], 0.0), "attention firt column should be 0"


def test_call(mha):
    Q = tf.random.normal((32, 100, 512))
    K = tf.random.normal((32, 100, 512))
    V = tf.random.normal((32, 100, 512))
    mask = tf.random.normal((32, 1, 1, 100)) > 0.0
    # errors
    K_wrong = tf.random.normal((31, 100, 512))
    with pytest.raises(ValueError, match=".*32.*"):
        mha(Q, K_wrong, V, mask)
    V_wrong = tf.random.normal((31, 100, 512))
    with pytest.raises(ValueError, match=".*32.*"):
        mha(Q, K, V_wrong, mask)
    # normal usage
    out, att = mha(Q, K, V, mask)
    assert out.shape == tf.TensorShape((32, 100, 512)), (
        "output of Multi-Heads Attention block has not the expected shape"
        )
    assert att.shape == tf.TensorShape((32, 8, 100, 100)), (
        "attention of Multi-Heads Attention block has not the expected shape"
        )
