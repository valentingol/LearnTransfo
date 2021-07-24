import pytest
import tensorflow as tf

from transformer.architecture.attention import MultiHeadAttention
from transformer.architecture.encoder import EncoderBlock, Encoder
from transformer.architecture.feedforward import FeedForward

@pytest.fixture(scope="module")
def encoder_block():
    return EncoderBlock(d_model=512, n_heads=8, d_ff=1024, dropout=0.2)

@pytest.fixture(scope="module")
def encoder():
    return Encoder(n_blocks=6, d_model=512, n_heads=8, d_ff=1024, dropout=0.2)


## EncoderBlock
def test_init_encoderblock():
    encoder_block = EncoderBlock(d_model=512, n_heads=8, d_ff=1024,
                                 dropout=0.2)
    assert encoder_block.d_model == 512, ("dropout d_model does not match "
                                          "the expected value")
    assert encoder_block.n_heads == 8, ("dropout n_heads does not match "
                                          "the expected value")
    assert encoder_block.dropout == 0.2, ("dropout attribute does not match "
                                          "the expected value")
    assert isinstance(encoder_block.mha, MultiHeadAttention), (
        "mha attribute should be an instance of MultiHeadAttention"
        )
    assert isinstance(encoder_block.ff, FeedForward), (
        "ff attribute should be an instance of FeedForward"
        )


def test_call_encoderblock(encoder_block):
    # errors
    inputs_wrong = tf.random.normal((32, 100, 511))
    with pytest.raises(ValueError, match='.*512.*'):
        encoder_block(inputs_wrong)
    # normal usage
    inputs = tf.random.normal((32, 100, 512))
    in_pad_mask = tf.random.normal((32, 1, 1, 100)) > 0
    block_outputs, attention = encoder_block(inputs, in_pad_mask=in_pad_mask,
                                      training=True)
    assert block_outputs.shape == tf.TensorShape((32, 100, 512)), (
        "block output of encoder block has not the expected shape"
        )
    assert attention.shape == tf.TensorShape((32, 8, 100, 100)), (
        "attention of encoder block has not the expected shape"
        )


## Encoder
def test_init_encoder():
    encoder = Encoder(n_blocks=6, d_model=512, n_heads=8, d_ff=1024,
                      dropout=0.2)
    assert encoder.n_blocks == 6, ("n_blocks attribute does not match the "
                               "expected value")
    assert encoder.d_model == 512, ("d_model attribute does not match the "
                               "expected value")
    assert encoder.n_heads == 8, ("n_heads attribute does not match the "
                             "expected value")
    assert encoder.dropout == 0.2, ("dropout attribute does not match the "
                                     "expected value")
    assert isinstance(encoder.encoder_blocks, list), (
        "encoder_blocks attribute should be a list"
        )
    assert len(encoder.encoder_blocks) == 6, ("encoder_blocks attribute "
                                              "length does not match the  "
                                              "expected value")
    for encoder_block in encoder.encoder_blocks:
        assert isinstance(encoder_block, EncoderBlock), (
            "values of encoder_blocks should be instances of EncoderBlock"
            )


def test_call_encoder(encoder):
    inputs = tf.random.normal((32, 100, 512))
    in_pad_mask = tf.random.normal((32, 1, 1, 100)) > 0
    encoder_outputs, attentions = encoder(inputs, in_pad_mask=in_pad_mask,
                                         training=True)
    assert encoder_outputs.shape == inputs.shape, ("the output shape do not "
                                                   "match the input shape")
    assert len(attentions) == 6, ("attentions of encoder has not the expected"
                                  " length")
    for i in range(6):
        assert attentions[i].shape == tf.TensorShape((32, 8, 100, 100)), (
            "attentions tensors of encoder have not the expected shape"
            )
