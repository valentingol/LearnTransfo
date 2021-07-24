import pytest
import tensorflow as tf

from tensorflow.keras.layers import Embedding
from transformer.text.embedding import TextEmbedding
from tests.tests_utils import tf_equal

@pytest.fixture(scope="module")
def emb():
    weights =  tf.Variable([[0.2, 0.7, 0.7, 0. , 0.4, 0.3, 0.6, 0.6],
                            [0.3, 0.2, 0.4, 0.4, 0.7, 0.9, 1. , 0.4],
                            [0.1, 0.1, 0.5, 0.3, 0.8, 0.6, 0.6, 0.6],
                            [0.9, 0. , 0.9, 0.6, 0.7, 0.7, 1. , 0.6],
                            [0.1, 0.1, 0.5, 0.4, 0.2, 0.4, 0.2, 0.7]])
    emb = TextEmbedding(vocab_size=5, depth=8, reverse_method='cosine')
    emb.set_weights([weights.numpy()])
    return emb

@pytest.fixture(scope="module")
def weights():
    return tf.Variable([[0.2, 0.7, 0.7, 0. , 0.4, 0.3, 0.6, 0.6],
                        [0.3, 0.2, 0.4, 0.4, 0.7, 0.9, 1. , 0.4],
                        [0.1, 0.1, 0.5, 0.3, 0.8, 0.6, 0.6, 0.6],
                        [0.9, 0. , 0.9, 0.6, 0.7, 0.7, 1. , 0.6],
                        [0.1, 0.1, 0.5, 0.4, 0.2, 0.4, 0.2, 0.7]])


# TextEmbedding
def test_init():
    # errors
    with pytest.raises(ValueError, match='.*__unknown__.*'):
        emb = TextEmbedding(vocab_size=2000, depth=1024,
                            reverse_method="__unknown__")
    emb = TextEmbedding(vocab_size=2000, depth=1024, reverse_method='classic')
    assert emb.vocab_size == 2000, ("vocab_size attribute does not match the "
                                    "expected value")
    assert emb.depth == 1024, ("depth attribute does not match the "
                                    "expected value")
    assert emb.reverse_method == 'classic', ("reverse_method attribute does "
                                             "not match the expected value")
    assert isinstance(emb.embedding, Embedding), (
        "embedding attribute is not an instace of the expected class"
        )
    assert emb.embedding.get_weights() != [], ("embedding not built at "
                                               "initialization")
    assert emb.embedding.get_weights()[0].shape == (2000, 1024), (
        "embedding weights have not match the expected shape"
        )


def test_call(emb, weights):
    res = emb(tf.constant([[0, 1, 2, 3, 4]]))
    assert res.shape == tf.TensorShape((1, 5, 8)), ("output shape does not "
                                                    "match the expected value")
    assert tf_equal(tf.squeeze(res), weights), ("output weights have not the "
                                                "expected values")
    res = emb(tf.random.uniform((6, 7, 8, 10), 0, 5, dtype=tf.int32))
    assert res.shape == tf.TensorShape((6, 7, 8, 10, 8)), ("output shape does "
                                                "not match the expected shape")


def test_reverse(emb):
    # errors
    emb.reverse_method = "__unknown__"
    features = tf.random.uniform((3, 8))
    with pytest.raises(ValueError, match='.*__unknown__.*'):
        emb.reverse(features)

    for reverse_method in ['classic', 'square_norm', 'distance', 'cosine']:
        emb.reverse_method = reverse_method
        features = tf.constant([[0.1, 0.1, 0.5, 0.3, 0.8, 0.6, 0.6, 0.6]])
        res = emb.reverse(features)
        assert res.shape == tf.TensorShape((1, 5)), ("output shape does not "
                                                    "match the expected shape")

        features = tf.random.uniform((6, 7, 8)) * 10
        res = emb.reverse(features)
        for x in tf.reshape(res, (-1, 1)):
            assert 0.0 <= x <= 1.0, "output probability is not in [0, 1]"
        assert tf_equal(tf.reduce_sum(res, axis=-1), 1.0), (
            "output probabilities are not normalized"
            )
        assert res.shape == tf.TensorShape((6, 7, 5)), (
            "output shape does not match the expected shape"
            )

    for reverse_method in ['square_norm', 'distance', 'cosine']:
        emb.reverse_method = reverse_method
        features = tf.constant([[0.1, 0.1, 0.5, 0.3, 0.8, 0.6, 0.6, 0.6]])
        res = emb.reverse(features)
        assert tf.argmax(res, axis=-1) == 2, ("output does not have the "
                                            "expected values")

        tokens = tf.random.uniform((6, 7, 8, 10), 0, 5, dtype=tf.int32)
        features = emb(tokens)
        res = emb.reverse(features)
        pred_tokens = tf.argmax(res, axis=-1)
        assert tf_equal(pred_tokens, tokens), ("predicted tokens does not "
                                            "match the input tokens")