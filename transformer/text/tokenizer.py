from os import remove
import re

import tensorflow as tf
import tensorflow_text as tft
from tensorflow_text.tools.wordpiece_vocab import (bert_vocab_from_dataset
                                                   as bert_vocab)


class TokenizerBert(object):
    """Implementation of the BERT Tokenizer (using subwords).
    """
    def __init__(self, max_vocab_size: int=8000, lower_case=True,
                 reserved_tokens: list[str]=["[PAD]", "[UNK]"],
                 remove_char: str="""#"$%&*+/<=>@]['_`{|}~\t\n"""):
        """
        Parameters
        ----------
        max_vocab_size : int, optional
            Maximum number of words in the vocabulary,
            by default 8000.
        lower_case : bool, optional
            Whether to lowercase the input text, by default True.
        reserved_tokens : list[str], optional
            List of reserved tokens, by default ["[PAD]", "[UNK]"]
        remove_char : str, optional
            Characters to be removed from text, by default:
            #"$%&*+/<=>@]['_`{|}~\t\n
        """
        self.max_vocab_size = max_vocab_size
        self.reserved_tokens = reserved_tokens
        self.bert_tokenizer_params = dict(lower_case=lower_case)
        self.vocab = None
        self.tokenizer = None
        if remove_char != "" :
            self.remove_char = [ '[' + c + ']' for c in remove_char]
        else:
            self.remove_char = None

    def build_tokenizer(self, dataset: tf.data.Dataset):
        """Build the tokenizer vocabulary from a dataset.

        Parameters
        ----------
        dataset : tf.data.Dataset
            Dataset from which the vocab will be built.
        """
        self.vocab = bert_vocab.bert_vocab_from_dataset(
                            dataset.prefetch(2),
                            vocab_size=self.max_vocab_size,
                            reserved_tokens=self.reserved_tokens,
                            bert_tokenizer_params=self.bert_tokenizer_params
                            )
        lookup_table = tf.lookup.StaticVocabularyTable(
                    tf.lookup.KeyValueTensorInitializer(
                    keys=self.vocab,
                    key_dtype=tf.string,
                    values=tf.range(
                        tf.size(self.vocab, out_type=tf.int64), dtype=tf.int64),
                    value_dtype=tf.int64),
                    num_oov_buckets=1
                )

        self.tokenizer = tft.BertTokenizer(lookup_table,
                                           **self.bert_tokenizer_params)

    def tokenize(self, text: tf.Tensor):
        """Transform text into a list of tokens.

        Parameters
        ----------
        text : tf.Tensor
            Tensor text to be tokenized.

        Raises
        ------
        ValueError :
            If the tokenizer is not built.

        Returns
        -------
        tokens : tf.Tensor
            Padded tensor containing tokens (padded tokens
            correspond to 0).
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer not built yet.")
        if text.ndim == 0:
                text = tf.expand_dims(text, 0)
        if self.remove_char is not None:
            text = text.numpy()
            text = [line.decode() for line in text]
            for c in self.remove_char:
                text = [re.sub(c, ' ', line) for line in text]
            text = tf.constant(text)
        tokens = self.tokenizer.tokenize(text)
        tokens = tokens.merge_dims(-2,-1)
        # add 0 to the end of sequences shorter than max_seq_length
        tokens = tokens.to_tensor()
        return tokens

    def detokenize(self, tokens: tf.Tensor):
        """Transform a list of tokens into a text tensor.

        Parameters
        ----------
        tokens : tf.Tensor
            Tensor containing tokens.

        Raises
        ------
        ValueError :
            If the tokenizer is not built.

        Returns
        -------
        text_tensor : tf.Tensor
            Tensor containing clean text.
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer not built yet.")
        tensor = self.tokenizer.detokenize(tokens)
        tensor = tf.where(tensor=='[PAD]', '', tensor)
        text_tensor = tf.strings.reduce_join(tensor, separator=' ', axis=-1)
        text_tensor = tf.strings.strip(text_tensor)
        return text_tensor

    def __call__(self, text):
        """Performes the tokenization (see .tokenize() method).
        """
        return self.tokenize(text)


if __name__ == '__main__':
    import os
    import sys
    myPath = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, myPath + '/../../')
    from datasets.scripts.fra_eng import datasets_fra_eng
    _, _, full_dataset = datasets_fra_eng()
    fr_dataset = full_dataset.map(lambda fr, _: fr)
    fr_tokenizer = TokenizerBert(lower_case=False)
    fr_tokenizer.build_tokenizer(fr_dataset.take(3000))
    fr_text = [
        "Nous sommes enthousiasmés par l'avenir des modèles basés",
        "sur l'attention et prévoyons de les appliquer à d'autres",
        "tâches. Nous prévoyons d'étendre les Transformers aux problèmes",
        "impliquant des modalités d'entrée et de sortie différentes",
        "de celles du texte, et d'étudier les mécanismes d'attention locale",
        "restreinte pour gérer efficacement des entrées et sorties",
        "volumineuses comme les images, l'audio et les vidéos. Rendre la",
        "génération moins séquentielle est un de nos autres objectifs",
        "de recherche."
        ]
    fr_dataset = tf.data.Dataset.from_tensor_slices(fr_text)
    print('vocab', fr_tokenizer.vocab)
    print('len vocab:', len(fr_tokenizer.vocab))
    print('\ntokens:')
    for text in fr_dataset.batch(2):
        tokens = fr_tokenizer.tokenize(text)
        print(tokens)
    print('\ntext:')
    for text in fr_dataset.batch(2):
        tokens = fr_tokenizer.tokenize(text)
        text_tensor = fr_tokenizer.detokenize(tokens)
        tf.print(text_tensor)
