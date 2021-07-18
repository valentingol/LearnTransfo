import os
import sys

import pytest
import tensorflow as tf
from tensorflow_text.python.ops.tokenization import Tokenizer

myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../../')

from tests.tests_utils import tf_equal
from transformer.text.tokenizer import TokenizerBert

@pytest.fixture(scope="module")
def dataset():
    text = [
        "To be, or not to be, that is the question:",
        "Whether 'tis nobler in the mind to suffer",
        "The slings and arrows of outrageous fortune,",
        "Or to take arms against a sea of troubles",
        "And by opposing end them.",
        ]
    dataset = tf.data.Dataset.from_tensor_slices(text)
    return dataset

@pytest.fixture(scope="module")
def text():
    return tf.constant([['the mind of the sea is', 'arrows against fortune,'],
                       ['I suffer the troubles', 'opposing to the sea.']])

@pytest.fixture(scope="module")
def token_bert():
    token_bert = TokenizerBert(max_vocab_size=8000, lower_case=True,
                              remove_char="'")
    return token_bert


## TokenizerBert
def test_init():
    token_bert = TokenizerBert(max_vocab_size=8000, lower_case=True,
                              reserved_tokens=['[PAD]', '[UNK]'],
                              remove_char="""#"$%&*+/<=>@]['_`{|}~\t\n""")
    assert token_bert.max_vocab_size == 8000, ("max_vocab_size attribute does "
                                             "not match the expected value")
    assert token_bert.bert_tokenizer_params == dict(lower_case=True), (
        "lower_case attribute does not match the expected value")
    assert token_bert.vocab is None, ( "vocab attribute does not match the "
                                    "expected value")
    assert token_bert.tokenizer is None, ( "vocab attribute does not match the "
                                    "expected value")
    assert token_bert.reserved_tokens == ['[PAD]', '[UNK]'], (
        "reserved_tokens attribute does not match the expected value"
        )
    remove_char = ['[#]', '["]', '[$]', '[%]', '[&]', '[*]', '[+]', '[/]',
                   '[<]', '[=]', '[>]', '[@]', '[]]', '[[]', "[']", '[_]',
                   '[`]', "[{]", '[|]', '[}]', '[~]', '[\t]', '[\n]']
    assert token_bert.remove_char == remove_char, (
        "remove_char attribute does not match the expected value"
        )


def test_build_tokenizer(dataset, token_bert):
    token_bert.build_tokenizer(dataset)
    for token in [ 'a', 'b', 'i', 't', '[PAD]', '[UNK]']:
        assert token in token_bert.vocab, (
            "vocab does not contain the token: {}".format(token))
    assert token_bert.tokenizer is not None, "tokenizer does not be None"
