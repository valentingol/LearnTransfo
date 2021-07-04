import tensorflow as tf

from transformer.architecture.attention import MultiHeadAttention
from transformer.architecture.feedforward import FeedForward

kl = tf.keras.layers

"""Computes the encoder found in the original transformer
implementation (Vaswani et al, 2017).
"""

class EncoderBlock(kl.Layer):

    def __init__(self, d_model: int, n_heads: int, d_ff: int,
                 dropout: float=0.0):
        """
        Parameters
        ----------
        d_model : int
            Depth of the model.
        n_heads : int
            Number of attention heads in the model.
        d_ff : int
            Hidden layer size of the feed forward network.
        dropout : float, optional
            Rate of dropout, by default 0.0.
        """
        super(EncoderBlock, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.dropout = dropout
        self.mha = MultiHeadAttention(d_model=d_model, n_heads=n_heads)
        self.ff = FeedForward(d_model=d_model, d_ff=d_ff)

    def __call__(self, inputs: tf.Tensor, in_pad_mask: tf.Tensor=None,
                 training: bool=False):
        """Returns the output of the encoder layer.

        Parameters
        ----------
        inputs : tf tensor, shape=(batch_size, seq_len, d_model)
            Input data tensor (full depth).
        in_pad_mask : None or tf tensor, optional
            Padding mask corresponding to input data. If None,
            no mask is applied, by default None.
        training : bool, optional
            Whether the algorithm is training or not,
            by default False.

        Raises
        ------
        ValueError:
            Last dimension of inputs is not self.d_model.

        Returns
        -------
        block_outputs : tf tensor
            shape=(batch_size, seq_len, d_model)
            Output of the encoder layer.
        attention : tf tensor
            shape=(batch_size, n_heads, seq_len, seq_len)
            Batched attention weights of the Multi-Head
            Attention block.
        """
        X = inputs
        if X.shape[-1] != self.d_model:
            raise ValueError("Input data tensor last dimension and specified "
                             f"d_model do not match (found {X.shape[-1]} and "
                             f"{self.d_model}")

        Layer_norm = kl.LayerNormalization
        # self attention
        att_outputs, attention = self.mha(X, X, X, mask=in_pad_mask)
        att_outputs = kl.Dropout(self.dropout)(att_outputs, training=training)
        att_outputs = Layer_norm(epsilon=1e-6)(att_outputs + X)

        ff_outputs = self.ff(att_outputs)
        ff_outputs = kl.Dropout(self.dropout)(ff_outputs, training=training)
        block_outputs = Layer_norm(epsilon=1e-6)(ff_outputs + att_outputs)
        # size=(batch_size, seq_len, d_model)

        return block_outputs, attention


class Encoder(kl.Layer):

    def __init__(self, n_blocks: int, d_model: int, n_heads: int, d_ff: int,
                 dropout=0.0):
        """
        Parameters
        ----------
        n_blocks : int
            Number of encoder blocks in the encoder.
        d_model : int
            Depth of the model.
        n_heads : int
            Number of attention heads in the model.
        d_ff : int
            Hidden layer size of the feed forward network.
        dropout : float, optional
            Rate of dropout, by default 0.0.
        """
        super(Encoder, self).__init__()
        self.n_blocks = n_blocks
        self.d_model = d_model
        self.n_heads = n_heads
        self.dropout = dropout
        self.encoder_blocks = [
            EncoderBlock(d_model=d_model, n_heads=n_heads, d_ff=d_ff,
                         dropout=dropout) for _ in range(n_blocks)
            ]

    def __call__(self, inputs: tf.Tensor, in_pad_mask: tf.Tensor=None,
                 training: bool=False):
        """Returns the output of the encoder.

        Parameters
        ----------
        inputs : tf tensor, shape=(batch_size, seq_len, d_model)
            Input data tensor (full depth).
        in_pad_mask : None or tf tensor, optional
            Padding mask corresponding to input data. If None,
            no mask is applied, by default None.
        training : bool, optional
            Whether the algorithm is training or not,
            by default False.

        Returns
        -------
        encoder_outputs : tf tensor
            shape=(batch_size, seq_len, d_model)
            Output of the encoder.
        attentions : list of tf tensors
            length=n_blocks
            tensor shapes=(batch_size, n_heads, seq_len, seq_len)
            Batched attention weights of Multi-Head Attention
            of all blocks.
        """
        X = inputs
        attentions = []
        X = kl.Dropout(self.dropout)(X, training=training)
        for encoder_block in self.encoder_blocks:
            X, attention = encoder_block(X, in_pad_mask=in_pad_mask,
                                         training=training)
            attentions.append(attention)
        encoder_outputs = X # (batch_size, seq_len, d_model)
        return encoder_outputs, attentions
