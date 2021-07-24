import tensorflow as tf
from tensorflow.python.keras.backend import arange

kl = tf.keras.layers

class TextEmbedding(kl.Layer):

    def __init__(self, vocab_size: int, depth: int, reverse_method='cosine'):
        """
        Parameters
        ----------
        vocab_size : int
            Size of input vocabulary.
        depth : int
            Depth of the embedding.
        method : str, optional
            One of 'classic', 'square_norm', 'distance', 'cosine'.
            Method use for reversing the embedding,
            by default 'cosine'.

        Raises
        ------
        ValueError :
            If `reverse_method` is unknown (not one of 'classic',
            'square_norm', 'distance' or 'cosine').
        """
        super(TextEmbedding, self).__init__()
        self.vocab_size = vocab_size
        self.depth = depth
        self.embedding = kl.Embedding(input_dim=self.vocab_size,
                                      output_dim=self.depth)
        self.embedding.build(vocab_size)
        if reverse_method in ('classic', 'norm', 'square_norm', 'distance',
                              'cosine'):
            self.reverse_method = reverse_method
        else:
            raise ValueError(f"Unknown method: {reverse_method}, should be one"
                             " of 'classic', 'square_norm', 'distance',"
                             "'cosine'.")

    def __call__(self, tokens):
        """Pass forward the embedding layer."""
        embeddings = self.embedding(tokens)
        return embeddings

    def reverse(self, features: tf.Tensor):
        """Reverse the embedding layer using self.reverse_method
        (used for sharing weights with the input embedding).

        Parameters
        ----------
        features : tf.Tensor, shape=(..., depth)
            Embedded features.

        Raises
        ------
        ValueError :
            If `reverse_method` is unknown (not one of 'classic',
            'square_norm', 'distance' or 'cosine').

        Returns
        -------
        probas : tf.Tensor, shape=(..., vocab_size)
            Probability distribution over the vocabulary.
        """
        # W: weights of the embedding layer
        W = self.embedding(tf.range(self.vocab_size))
        if self.reverse_method == 'classic':
            scores = tf.matmul(features, W, transpose_b=True)
        elif self.reverse_method == 'square_norm':
            square_norms = tf.norm(W, axis=-1, keepdims=True) * 2 + 1e-6
            W = W / square_norms
            scores = tf.matmul(features, W, transpose_b=True)
        elif self.reverse_method == 'distance':
            square_norms = tf.norm(W, axis=-1, keepdims=False) * 2 + 1e-6
            biais = - square_norms / 2
            scores = tf.matmul(features, W, transpose_b=True) + biais
        elif self.reverse_method == 'cosine':
            norms = tf.norm(W, axis=-1, keepdims=True) + 1e-6
            W = W / norms
            scores = tf.matmul(features, W, transpose_b=True)
        else:
            raise ValueError(f"Unknown method: {self.reverse_method}, should "
                             "be one of 'classic', 'square_norm', 'distance',"
                             "'cosine'.")
        probas = tf.nn.softmax(scores, axis=-1)
        return probas
