import tensorflow as tf

from transformer.architecture.attention import MultiHeadAttention
from transformer.architecture.feedforward import FeedForward

kl = tf.keras.layers

"""Computes the decoder found in the original transformer
implementation (Vaswani et al, 2017).
"""

class DecoderBlock(kl.Layer):

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

        super(DecoderBlock, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.dropout = dropout
        self.mha_self = MultiHeadAttention(d_model=d_model, n_heads=n_heads)
        self.mha_encoder = MultiHeadAttention(d_model=d_model, n_heads=n_heads)
        self.ff = FeedForward(d_model=d_model, d_ff=d_ff)
        self.layernorm1 = Layer_norm(epsilon=1e-6)
        self.layernorm2 = Layer_norm(epsilon=1e-6)
        self.layernorm3 = Layer_norm(epsilon=1e-6)
        self.dropoutlayer = kl.Dropout(rate=dropout)

    def __call__(self, outputs: tf.Tensor, encoder_outputs: tf.Tensor,
                 out_mask: tf.Tensor, in_pad_mask: tf.Tensor=None,
                 training: bool=False):
        """Returns the output of the decoder layer.

        Parameters
        ----------
        outputs : tf tensor, shape=(batch_size, seq_len, d_model)
            Output data tensor (full depth).
        encoder_outputs : tf tensor, shape=(batch_size, seq_len, d_model)
            Output of the encoder.
        out_mask : tf tensor
            Mask corresponding to output data. It combines the
            padding mask of output data and the future mask.
        in_pad_mask : None or tf tensor, optional
            Padding mask corresponding to input (encoder) data.
            If None, no mask is applied, by default None.
        training : bool, optional
            Whether the algorithm is training or not,
            by default False.

        Raises
        ------
        ValueError:
            Last dimensions of outputs is not self.d_model.

        Returns
        -------
        block_outputs : tf tensor
            shape=(batch_size, seq_len, d_model)
            Output of the encoder layer.
        self_attention : tf tensor
            shape=(batch_size, n_heads, seq_len, seq_len)
            Batched attention weights of the first Multi-Head
            Attention block (self attention).
        encoder_attention : tf tensor
            shape=(batch_size, n_heads, seq_len, seq_len)
            Batched attention weights of the second Multi-Head
            Attention block (attention based on encoder output).
        """
        if outputs.get_shape()[-1] != self.d_model:
            raise ValueError("Last dimension of 'outputs' should be equal to "
                             f"d_model, found {outputs.get_shape()[-1]} and "
                             f"{self.d_model}.")
        X = outputs
        E = encoder_outputs
        # self attention
        X_norm = self.layernorm1(X)
        pre_X1, self_attention = self.mha_self(X_norm, X_norm, X_norm,
                                                     mask=out_mask)
        pre_X1 = self.dropoutlayer(pre_X1, training=training)
        X1 = pre_X1 + X # size=(batch_size, seq_len, d_model)
        # encoder output attention
        X1_norm = self.layernorm2(X1)
        pre_X2, encoder_attention = self.mha_encoder(X1_norm, E, E,
                                                           mask=in_pad_mask)
        pre_X2 = self.dropoutlayer(pre_X2, training=training)
        X2 = pre_X2 + X1 # size=(batch_size, seq_len, d_model)
        # feed forward
        X2_norm = self.layernorm3(X2)
        pre_X3 = self.ff(X2_norm)
        pre_X3 = self.dropoutlayer(pre_X3, training=training)
        X3 = pre_X3 + X2 # size=(batch_size, seq_len, d_model)
        block_outputs = X3
        return block_outputs, self_attention, encoder_attention


class Decoder(kl.Layer):

    def __init__(self, n_blocks: int, d_model: int, n_heads: int, d_ff: int,
                 dropout=0.0):
        """
        Parameters
        ----------
        n_blocks : int
            Number of decoder blocks in the decoder.
        d_model : int
            Depth of the model.
        n_heads : int
            Number of attention heads in the model.
        d_ff : int
            Hidden layer size of the feed forward network.
        dropout : float, optional
            Rate of dropout, by default 0.0.
        """
        super(Decoder, self).__init__()
        self.n_blocks = n_blocks
        self.d_model = d_model
        self.n_heads = n_heads
        self.dropout = dropout
        self.decoder_blocks = [
            DecoderBlock(d_model=d_model, n_heads=n_heads, d_ff=d_ff,
                         dropout=dropout) for _ in range(n_blocks)
            ]
        self.layernorm = kl.LayerNormalization(epsilon=1e-6)

    def __call__(self, outputs: tf.Tensor, encoder_outputs: tf.Tensor,
                 out_mask: tf.Tensor, in_pad_mask: tf.Tensor=None,
                 training: bool=False):
        """Returns the output of the decoder.

        Parameters
        ----------
        outputs : tf tensor, shape=(batch_size, seq_len, d_model)
            Output data tensor (full depth).
        out_mask : tf tensor
            Mask corresponding to output data. It combines the
            padding mask of output data and the future mask.
        in_pad_mask : None or tf tensor
            Padding mask corresponding to input data. If None,
            no mask is applied, by default None.
        training : bool, optional
            Whether the algorithm is training or not,
            by default False.

        Returns
        -------
        decoder_outputs : tf tensor
            shape=(batch_size, seq_len, d_model)
            Output of the decoder.
        self_attentions : list of tf tensors
            length=n_blocks
            tensor shapes=(batch_size, n_heads, seq_len, seq_len)
            Batched attention weights of the first Multi-Head
            Attention blocks (self attentions).
        encoder_attentions : list of tf tensors
            length=n_blocks
            tensor shapes=(batch_size, n_heads, seq_len, seq_len)
            Batched attention weights of the second Multi-Head
            Attention blocks (attentions based on encoder output).
        """
        X = outputs
        E = self.layernorm(encoder_outputs)
        self_attentions, encoder_attentions = [], []
        X = kl.Dropout(self.dropout)(X, training=training)
        for decoder_block in self.decoder_blocks:
            X, att1, att2 = decoder_block(X, encoder_outputs=E,
                                         out_mask=out_mask,
                                         in_pad_mask=in_pad_mask,
                                         training=training)
            self_attentions.append(att1)
            encoder_attentions.append(att2)
        decoder_outputs = X # (batch_size, seq_len, d_model)
        return decoder_outputs, self_attentions, encoder_attentions
