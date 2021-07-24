import pytest
import tensorflow as tf

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
    return tf.constant(['the mind of the sea is', 'arrows against fortune,',
                       'I suffer the troubles', 'opposing to the sea.'])


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
    # when no removing char
    token_bert = TokenizerBert(max_vocab_size=8000, lower_case=True,
                               remove_char="")
    assert token_bert.remove_char is None, ("remove_char attribute does not "
                                            "match the expected value")


def test_build_tokenizer(dataset):
    token_bert = TokenizerBert(max_vocab_size=8000, lower_case=True)
    token_bert.build_tokenizer(dataset)
    for token in [ 'a', 'b', 'i', 't', '[PAD]', '[UNK]']:
        assert token in token_bert.vocab, (
            "vocab does not contain the token: {}".format(token))
    assert token_bert.tokenizer is not None, "tokenizer does not be None"


def test_tokenize(text, dataset):
    token_bert = TokenizerBert(max_vocab_size=8000, lower_case=True)
    # error
    with pytest.raises(ValueError, match="Tokenizer not built.*"):
        token_bert.tokenize(text)
    token_bert.build_tokenizer(dataset)
    wrong_text = tf.constant([['the mind of the sea is',
                               'arrows against fortune,'],
                              ['I suffer the troubles',
                               'opposing to the sea.']])
    with pytest.raises(ValueError, match=".*2-D tensor.*"):
        token_bert.tokenize(wrong_text)
    # normal use
    tokens = token_bert.tokenize(text)
    assert isinstance(tokens, tf.Tensor), "tokens should be a tf tensor"
    assert tokens.dtype == tf.int64, "tokens tye should be tf.int64"
    assert tf.rank(tokens) == 2, ("tokens rank does not match the expected "
                                  "value")
    assert tokens.shape[0] == 4, (
        "tokens first dimension does not match the expected value"
        )
    # when no removing char
    token_bert = TokenizerBert(max_vocab_size=8000, lower_case=True,
                               remove_char="")
    token_bert.build_tokenizer(dataset)
    token_bert.tokenize(text)
    # when rank(text) == 0
    text_0 = tf.constant('the mind of the sea is')
    text_tensor = token_bert.tokenize(text_0)
    assert tf.rank(text_tensor) == 2, "text_tensor should be a 2-D tensor"


def test_call(text, dataset):
    token_bert = TokenizerBert(max_vocab_size=8000, lower_case=True)
    token_bert.build_tokenizer(dataset)
    tokens1 = token_bert.tokenize(text)
    tokens2 = token_bert(text)
    assert tf_equal(tokens1, tokens2), ("tokens output of __call__ method"
                                        "does notmatch the expected value")


def test_detokenize(text, dataset):
    token_bert = TokenizerBert(max_vocab_size=8000, lower_case=True)
    # error
    tokens = tf.random.uniform((6, 10), 0, 10, dtype=tf.int64)
    with pytest.raises(ValueError, match="Tokenizer not built.*"):
        token_bert.detokenize(tokens)
    token_bert.build_tokenizer(dataset)
    text = token_bert.detokenize(tokens)
    assert isinstance(text, tf.Tensor), "text should be a tf tensor"
    assert len(text) == 6, "text should have 4 elements"
    for i in range(6):
        assert type(text[i].numpy()) == bytes, (f"text element {i} should be "
                                        "a bytes string")
