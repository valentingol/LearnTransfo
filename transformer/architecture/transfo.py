import tensorflow as tf

from transformer.architecture.decoder import Decoder
from transformer.architecture.encoder import Encoder
from transformer.architecture.pos_encoding import PostionalEncoding
from transformer.mask.mask import get_masks
from transformer.text.embedding import TextEmbedding


class BaseTransformerAutoEncoder(tf.keras.Model):
    """Base class for the Transformer model (Auto-Encoder).
    """
    def __init__(self, n_layers: int, d_model: int, n_heads: int, d_ff: int,
                dropout: float=0.1):
        """
        Parameters
        ----------
        n_layers : int
            Number of encoder and decoder layers/blocks.
        d_model : int
            Depth of the model.
        n_heads : int
            Number of attention heads.
        d_ff : int
            Hidden layer size of the feed forward network.
        dropout : float, optional
            Rate of dropout, by default 0.0.
        """
        super(BaseTransformerAutoEncoder, self).__init__()
        self.n_layers = n_layers
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.dropout = dropout
        self.encoder = Encoder(n_blocks=n_layers, d_model=d_model,
                               n_heads=n_heads, d_ff=d_ff, dropout=dropout)

        self.decoder = Decoder(n_blocks=n_layers, d_model=d_model,
                               n_heads=n_heads, d_ff=d_ff, dropout=dropout)

    def __call__(self, inputs: tf.Tensor, outputs: tf.Tensor,
             out_mask: tf.Tensor,in_pad_mask: tf.Tensor=None,
             training: bool=False):
        """Forward pass of Base Auto-Encoder Transformer.

        Parameters
        ----------
        inputs : tf tensor, shape=(batch_size, seq_len, d_model)
            Input data tensor (full depth).
        outputs : tf tensor, shape=(batch_size, seq_len, d_model)
            Output data tensor (full depth).
        out_mask : tf tensor
            Mask corresponding to output data. It combines the
            padding mask of output data and the future mask.
        in_pad_mask : None or tf tensor, optional
            Padding mask corresponding to input data. If None,
            no mask is applied, by default None.
        training : bool, optional
            Whether the algorithm is training or not,
            by default False.

        Returns
        -------
        decoder_outputs: tf.tensor
            shape=(batch_size, seq_len, d_model)
            Output of the decoder.
        attentions: dict
            Dictionary of attention weights from:
            encoder, decoder (self) and decoder (with encoder).
        """
        encoder_outputs, enc_attention = self.encoder(
            inputs, in_pad_mask, training
            )
        # shape=(batch_size, inputs_seq_len, d_model)

        decoder_outputs, dec_self_attention, dec_enc_attention = self.decoder(
            outputs, encoder_outputs, out_mask, in_pad_mask, training
            )
        # shape=(batch_size, outputs_seq_len, d_model)

        attentions = {'encoder': enc_attention,
                      'decoder_self': dec_self_attention,
                      'decoder_encoder': dec_enc_attention}

        return decoder_outputs, attentions


class TransformerNLP(BaseTransformerAutoEncoder):
    def __init__(self, n_layers: int, d_model: int, n_heads: int, d_ff: int,
                 dropout: float, max_seq_len: int=None,
                 in_vocab_size: int=None,
                 out_vocab_size: int=None,
                 final_layer: str='output_embedding',
                 positional_encoding: PostionalEncoding=None,
                 input_embedding: TextEmbedding=None,
                 output_embedding: TextEmbedding=None,
                ):
        if final_layer not in ['output_embedding', 'dense']:
            raise ValueError(f"Final layer option '{final_layer}' not "
                             "understood should be one of: 'output_embedding',"
                             " 'dense'")
        if positional_encoding is None:
            if max_seq_len is None:
                print('FutureWarning: max_seq_len and positional_encoding are '
                      'not specified, the maximum seq len is automatically set'
                      ' to 150 to create positional_encoding. This could raise'
                      ' an error is some of your sequences are longer. To '
                      'avoid that, please set the positional encoding or the '
                      'maximum seq len.')
                positional_encoding = PostionalEncoding(150, d_model)
            else:
                positional_encoding = PostionalEncoding(max_seq_len, d_model)
        if input_embedding is None:
            input_embedding = TextEmbedding(in_vocab_size, d_model)
        if output_embedding is None:
            output_embedding = TextEmbedding(out_vocab_size, d_model)

        super(TransformerNLP, self).__init__(n_layers=n_layers,
                                             d_model=d_model,
                                             n_heads=n_heads,
                                             d_ff=d_ff,
                                             dropout=dropout)
        self.max_seq_len = max_seq_len
        self.positional_encoding = positional_encoding
        self.input_embedding = input_embedding
        self.output_embedding = output_embedding

        if final_layer == 'dense':
            self.final_layer = tf.keras.layers.Dense(out_vocab_size,
                                                 activation='softmax')
        else:
            self.final_layer = None

    def __call__(self, inputs_tokens: tf.Tensor, outputs_tokens: tf.Tensor,
                 training: bool=True):
        """Forward pass of NLP Transformer.

        Parameters
        ----------
        inputs_tokens: int tf.Tensor
            Inputstokens tensor.
        outputs_tokens: int tf.Tensor
            Output tokens tensor.
        training : bool, optional
            Whether the algorithm is training or not,
            by default True.

        Returns
        -------
        proba_output: tf.Tensor, shape=(batch_size, len_out_vocab)
            Output probabilities of the Transformer
            for each output token.
        attentions: dict
            Dictionary of attention weights from:
            encoder, decoder (self) and decoder (with encoder).
        """
        if self.max_seq_len is not None:
            # cut to max sequence length if longer
            inputs_tokens = inputs_tokens[..., :self.max_seq_len]
            outputs_tokens = outputs_tokens[..., :self.max_seq_len]
        # embeddings
        inputs = self.input_embedding(inputs_tokens)
        outputs = self.output_embedding(outputs_tokens)
        # first normalization
        inputs *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        outputs *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        # positional encoding
        pos_code = self.positional_encoding()
        inputs += pos_code[..., :tf.shape(inputs)[-2], :]
        outputs += pos_code[..., :tf.shape(outputs)[-2], :]
        # masks
        in_pad_mask, out_mask = get_masks(inputs_tokens, outputs_tokens)
        # forward pass
        decoder_outputs, attentions = super(TransformerNLP, self).__call__(
            inputs=inputs, outputs=outputs, out_mask=out_mask,
            in_pad_mask=in_pad_mask, training=training
            )
        # decode embedding
        if self.final_layer is None:
            proba_output = self.output_embedding.reverse(decoder_outputs)
        else:
            proba_output = self.final_layer(decoder_outputs)

        return proba_output, attentions

    def predict(self, *args, **kwargs):
        raise NotImplementedError("Not implemented yet.")
