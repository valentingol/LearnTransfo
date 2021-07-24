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
        Layer_norm = kl.LayerNormalization

        super(EncoderBlock, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.dropout = dropout
        self.mha = MultiHeadAttention(d_model=d_model, n_heads=n_heads)
        self.ff = FeedForward(d_model=d_model, d_ff=d_ff)
        self.layernorm1 = Layer_norm(epsilon=1e-6)
        self.layernorm2 = Layer_norm(epsilon=1e-6)
        self.dropoutlayer = kl.Dropout(rate=dropout)

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
        if inputs.get_shape()[-1] != self.d_model:
            raise ValueError("Last dimension of 'inputs' should be equal to "
                             f"d_model, found {inputs.get_shape()[-1]} and "
                             f"{self.d_model}.")
        X = inputs
        # self attention
        X_norm = self.layernorm1(X)
        pre_X1, attention = self.mha(X_norm, X_norm, X_norm, mask=in_pad_mask)
        pre_X1 = self.dropoutlayer(pre_X1, training=training)
        X1 = pre_X1 + X # size=(batch_size, seq_len, d_model)
        # feed forward
        X1_norm = self.layernorm2(X1)
        pre_X2 = self.ff(X1_norm)
        pre_X2 = self.dropoutlayer(pre_X2, training=training)
        X2 = pre_X2 + X1 # size=(batch_size, seq_len, d_model)
        block_outputs = X2
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
