import os
import sys

import pytest
import tensorflow as tf

myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../../')

from transformer.architecture.attention import MultiHeadAttention
from transformer.architecture.decoder import DecoderBlock, Decoder
from transformer.architecture.feedforward import FeedForward

@pytest.fixture
def decoder_block():
    return DecoderBlock(d_model=512, n_heads=8, d_ff=1024, dropout=0.2)

@pytest.fixture
def decoder():
    return Decoder(n_blocks=6, d_model=512, n_heads=8, d_ff=1024, dropout=0.2)


## DecoderBlock
def test_init_decoderblock():
    decoder_block = DecoderBlock(d_model=512, n_heads=8, d_ff=1024,
                                 dropout=0.2)
    assert decoder_block.d_model == 512, ("dropout d_model does not match "
                                          "the expected value")
    assert decoder_block.n_heads == 8, ("dropout n_heads does not match "
                                          "the expected value")
    assert decoder_block.dropout == 0.2, ("dropout attribute does not match "
                                          "the expected value")
    assert isinstance(decoder_block.mha_self, MultiHeadAttention), (
        "mha_self attribute should be an instance of MultiHeadAttention"
        )
    assert isinstance(decoder_block.mha_encoder, MultiHeadAttention), (
        "mha_encoder attribute should be an instance of MultiHeadAttention"
        )
    assert isinstance(decoder_block.ff, FeedForward), (
        "ff attribute should be an instance of FeedForward"
        )


def test_call_decoderblock(decoder_block):
    outputs = tf.random.normal((32, 100, 512))
    encoder_outputs = tf.random.normal((32, 100, 512))
    out_mask = tf.random.normal((32, 1, 1, 100)) > 0
    in_pad_mask = tf.random.normal((32, 1, 1, 100)) > 0
    # errors
    outputs_wrong = tf.random.normal((32, 100, 511))
    with pytest.raises(ValueError, match='.*512.*'):
        decoder_block(outputs_wrong, encoder_outputs=encoder_outputs,
                      out_mask=out_mask)
    # normal usage
    block_outputs, att1, att2 = decoder_block(outputs,
                                             encoder_outputs=encoder_outputs,
                                             out_mask=out_mask,
                                             in_pad_mask=in_pad_mask,
                                             training=True)
    assert block_outputs.shape == tf.TensorShape((32, 100, 512)), (
        "block output of decoder block has not the expected shape"
        )
    assert att1.shape == tf.TensorShape((32, 8, 100, 100)), (
        "self_attention of decoder block has not the expected shape"
        )
    assert att2.shape == tf.TensorShape((32, 8, 100, 100)), (
        "encoder_attention of decoder block has not the expected shape"
        )


## Decoder
def test_init_decoder():
    decoder = Decoder(n_blocks=6, d_model=512, n_heads=8, d_ff=1024,
                      dropout=0.2)
    assert decoder.n_blocks == 6, ("n_blocks attribute does not match the "
                               "expected value")
    assert decoder.d_model == 512, ("d_model attribute does not match the "
                               "expected value")
    assert decoder.n_heads == 8, ("n_heads attribute does not match the "
                             "expected value")
    assert decoder.dropout == 0.2, ("dropout attribute does not match the "
                                     "expected value")
    assert isinstance(decoder.decoder_blocks, list), (
        "decoder_blocks attribute should be a list"
        )
    assert len(decoder.decoder_blocks) == 6, ("decoder_blocks attribute "
                                              "length does not match the  "
                                              "expected value")
    for encoder_block in decoder.decoder_blocks:
        assert isinstance(encoder_block, DecoderBlock), (
            "values of decoder_blocks should be instances of DecoderBlock"
            )


def test_call_decoder(decoder):
    outputs = tf.random.normal((32, 100, 512))
    encoder_outputs = tf.random.normal((32, 100, 512))
    out_mask = tf.random.normal((32, 1, 1, 100)) > 0
    in_pad_mask = tf.random.normal((32, 1, 1, 100)) > 0
    decoder_outputs, att1, att2 = decoder(outputs,
                                        encoder_outputs=encoder_outputs,
                                        out_mask=out_mask,
                                        in_pad_mask=in_pad_mask,
                                        training=True)
    assert decoder_outputs.shape == outputs.shape, (
        "the output shape ('decoder_outputs') do not match the input shape "
        "('outputs')")
    assert len(att1) == 6 and len(att2) == 6, (
        "self_attentions and encoder_attentions of decoder have not the "
        "expected length")
    for i in range(6):
        assert att1[i].shape == tf.TensorShape((32, 8, 100, 100)), (
            "self_attentions tensors of decoder have not the expected shape"
            )
        assert att2[i].shape == tf.TensorShape((32, 8, 100, 100)), (
            "encoder_attentions tensors of decoder have not the expected shape"
            )
