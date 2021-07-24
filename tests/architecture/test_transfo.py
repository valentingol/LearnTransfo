import pytest
import tensorflow as tf

from tests.tests_utils import tf_equal
from transformer.architecture.decoder import Decoder
from transformer.architecture.encoder import Encoder
from transformer.architecture.pos_encoding import PostionalEncoding
from transformer.architecture.transfo import BaseTransformerAutoEncoder
from transformer.architecture.transfo import TransformerNLP
from transformer.text.embedding import TextEmbedding

@pytest.fixture(scope="module")
def btae():
    return BaseTransformerAutoEncoder(n_layers=6, d_model=512, n_heads=8,
                                      d_ff=2048, dropout=0.1)

@pytest.fixture(scope="module")
def tnlp():
    return TransformerNLP(n_layers=6, d_model=512, n_heads=8, d_ff=2048,
                          dropout=0.1, max_seq_len=100, in_vocab_size=50,
                          out_vocab_size=60)


# BaseTransformerAutoEncoder
def test_init_BTAE():
    btae = BaseTransformerAutoEncoder(n_layers=4, d_model=256, n_heads=4,
                                      d_ff=1024, dropout=0.2)
    assert btae.n_layers == 4, ("n_layers attribute does not match the "
                                    "expected value")
    assert btae.d_model == 256, ("d_model attribute does not match the "
                                    "expected value")
    assert btae.n_heads == 4, ("n_heads attribute does not match the "
                                    "expected value")
    assert btae.d_ff == 1024, ("d_ff attribute does not match the "
                                    "expected value")
    assert btae.dropout == 0.2, ("dropout attribute does not match the "
                                    "expected value")
    assert isinstance(btae.encoder, Encoder), (
        "encoder attribute is not an instance of the expected class"
        )
    assert isinstance(btae.decoder, Decoder), (
        "decoder attribute is not an instance of the expected class"
        )


def test_call_BTAE(btae):
    inputs = tf.random.uniform((32, 100, 512))
    outputs = tf.random.uniform((32, 150, 512))
    out_mask = tf.random.uniform((32, 1, 150, 150), 0, 2, dtype=tf.int32)
    in_pad_mask = tf.random.uniform((32, 1, 1, 100), 0, 2, dtype=tf.int32)
    dec_output, attention = btae(inputs=inputs, outputs=outputs,
                             out_mask=out_mask, in_pad_mask=in_pad_mask,
                             training=True)
    assert dec_output.shape == (32, 150, 512), ("dec_output does not have the "
                                                "expected shape")
    keys = {'encoder','decoder_self', 'decoder_encoder'}
    assert keys <= attention.keys(), ("attention dict does not contain all "
                                        "expected keys")


# TransformerNLP
def test_init_TNLP(capfd):
    # errors
    with pytest.raises(ValueError, match='.*__unknown__.*'):
        TransformerNLP(n_layers=4, d_model=256, n_heads=4, d_ff=1024,
                          dropout=0.2, max_seq_len=100, in_vocab_size=50,
                          out_vocab_size=60, final_layer='__unknown__')
    # normal use
    tnlp = TransformerNLP(n_layers=4, d_model=256, n_heads=4, d_ff=1024,
                          dropout=0.2, max_seq_len=100, in_vocab_size=50,
                          out_vocab_size=60, final_layer='dense')
    assert tnlp.n_layers == 4, ("n_layers attribute does not match the "
                                "expected value")
    assert tnlp.d_model == 256, ("d_model attribute does not match the "
                                "expected value")
    assert tnlp.n_heads == 4, ("n_heads attribute does not match the "
                                "expected value")
    assert tnlp.d_ff == 1024, ("d_ff attribute does not match the "
                                "expected value")
    assert tnlp.dropout == 0.2, ("dropout attribute does not match the "
                                "expected value")
    assert tnlp.max_seq_len == 100, ("max_seq_len attribute does not match the"
                                " expected value")
    assert isinstance(tnlp.encoder, Encoder), (
        "encoder attribute is not an instance of the expected class"
        )
    assert isinstance(tnlp.decoder, Decoder), (
        "decoder attribute is not an instance of the expected class"
        )
    assert isinstance(tnlp.input_embedding, TextEmbedding), (
        "input_embedding attribute is not an instance of the expected class"
        )
    assert isinstance(tnlp.output_embedding, TextEmbedding), (
        "output_embedding attribute is not an instance of the expected class"
        )
    assert isinstance(tnlp.final_layer, tf.keras.layers.Dense), (
        "final_layer attribute is not an instance of the expected class"
        )
    pos_encoding = PostionalEncoding(seq_len=100, depth=256)
    input_emb = TextEmbedding(vocab_size=50, depth=256)
    output_emb = TextEmbedding(vocab_size=60, depth=256)
    tnlp = TransformerNLP(n_layers=4, d_model=256, n_heads=4, d_ff=1024,
                          dropout=0.2, final_layer='output_embedding',
                          positional_encoding=pos_encoding,
                          input_embedding=input_emb,
                          output_embedding=output_emb)
    assert tnlp.final_layer is None, ("final_layer attribute does not "
                                "match the expected value")
    # case final_layer='output_embedding'
    tnlp = TransformerNLP(n_layers=4, d_model=256, n_heads=4, d_ff=1024,
                          dropout=0.2, final_layer='output_embedding',
                          input_embedding=input_emb,
                          output_embedding=output_emb)
    out, _ = capfd.readouterr()
    assert out.startswith('FutureWarning'), ("a futurewarning should be raise "
                                             "in these conditions")


def test_call_TNLP(tnlp):
    # case final_layer='output_embedding'
    inputs_tokens = tf.random.uniform((2, 40), 0, 50, dtype=tf.int32)
    outputs_tokens = tf.random.uniform((2, 38), 0, 60, dtype=tf.int32)
    proba, attentions = tnlp(inputs_tokens, outputs_tokens, training=True)
    keys = {'encoder','decoder_self', 'decoder_encoder'}
    assert keys <= attentions.keys(), ("attention dict does not contain all "
                                        "expected keys")
    assert proba.shape == tf.TensorShape((2, 38, 60)), (
        "proba does not have the expected shape"
        )
    for x in tf.reshape(proba, (-1, 1)):
            assert 0.0 <= x <= 1.0, "output probability is not in [0, 1]"
    assert tf_equal(tf.reduce_sum(proba, axis=-1), 1.0), (
        "output probabilities are not normalized"
        )
    # case final_layer == 'dense"
    tnlp = TransformerNLP(n_layers=4, d_model=256, n_heads=4, d_ff=1024,
                          dropout=0.2, max_seq_len=100, in_vocab_size=50,
                          out_vocab_size=60, final_layer='dense')
    proba, attentions = tnlp(inputs_tokens, outputs_tokens, training=True)
    assert proba.shape == tf.TensorShape((2, 38, 60)), (
        "proba does not have the expected shape"
        )
    for x in tf.reshape(proba, (-1, 1)):
            assert 0.0 <= x <= 1.0, "output probability is not in [0, 1]"
    assert tf_equal(tf.reduce_sum(proba, axis=-1), 1.0), (
        "output probabilities are not normalized"
        )


def test_predict_TNLP(tnlp):
    with pytest.raises(NotImplementedError):
        tnlp.predict()
