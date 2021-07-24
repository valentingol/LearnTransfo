import tensorflow as tf
kl = tf.keras.layers

class MultiHeadAttention(kl.Layer):
    def __init__(self, d_model: int, n_heads: int):
        """
        Parameters
        ----------
        d_model : int
            Depth of the model.
        n_heads : int
            Number of attention heads in the model.

        Raises
        ------
        ValueError
            n_heads does not divide d_model.
        """
        if d_model % n_heads != 0:
            raise ValueError(f"the number of heads ({n_heads}) does not divide"
                             f" the model depth({d_model})")

        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        # d_head: depth of each heads
        self.d_head = d_model // self.n_heads

        self.Wq = kl.Dense(d_model)
        self.Wk = kl.Dense(d_model)
        self.Wv = kl.Dense(d_model)
        self.Wo = kl.Dense(d_model)

    def _split_to_heads(self, X: tf.Tensor):
        """Splits the last dimension into (n_heads, d_head)
        and transposes output to have the heads dimension as
        second dimension.

        Parameters
        ----------
        X : tf tensor, shape=(batch_size, seq_len, d_model)
            Data (typically Q, K or V) with full depth.

        Raises
        ------
        ValueError:
            The last dimension of X does not match d_model.

        Returns
        -------
        X_split : tf tensor, shape=(batch_size, n_heads, seq_len, d_head)
            X splitted into heads. The first dimension is the dimension
            of batches, the second dimension is the dimension of heads.
        """
        if X.get_shape()[-1] != self.d_model:
            raise ValueError(f"the last dimension of X must be equal to "
                              f"d_model, found {X.get_shape()[-1]} and "
                              f"{self.d_model}.")
        batch_size = tf.shape(X)[0]
        X_split = tf.reshape(X, (batch_size, -1, self.n_heads, self.d_head))
        X_split = tf.transpose(X_split, perm=[0, 2, 1, 3])
        return X_split

    def scaled_dot_product_attention(self, Q: tf.Tensor, K: tf.Tensor,
                                     V: tf.Tensor, mask: tf.Tensor=None):
        """Returns scaled dot-product attention of the inputs.

        Parameters
        ----------
        Q : tf.tensor, shape=(..., lenQ, dQ)
            Queries.
        K : tf.tensor, shape=(..., lenK, dK)
            Keys, dK must be equal to dQ.
        V : tf.tensor, shape=(..., lenV, dV)
            Values, lenV must be equal to lenK.
        mask : None or tf.tensor, optional
            Mask to apply after the product Q @ K.T to avoid
            backpropagation on some data. Must be broadcastable with
            Q @ K.T. If None, no mask are used, by default None.

        Raises
        ------
        ValueError:
            The sizes of input tensors do not match.

        Returns
        -------
        output : tf.tensor, shape=(..., lenQ, dV)
            Results of scaled dot-product attention.
        attention : tf.tensor, shape = (..., lenQ, lenK)
            Attention weights (output of the softmax).
        """
        if Q.get_shape()[-1] != K.get_shape()[-1]:
            raise ValueError("the last dimension of Q and K must be equal, "
                             f"found {Q.get_shape()[-1]} and "
                             f"{K.get_shape()[-1]}.")
        if K.get_shape()[-2] != V.get_shape()[-2]:
            raise ValueError("the penultimate dimension of Q and K must be "
                             f"equal, found {K.get_shape()[-2]} and "
                             f"{V.get_shape()[-2]}.")
        QK = tf.matmul(Q, K, transpose_b = True) # (..., lenQ, lenK)
        dK = tf.cast(tf.shape(K)[-1], tf.float32)
        logits = QK / tf.math.sqrt(dK)
        if mask is not None:
            # cast mask to binary tensor (0.0 or 1.0)
            mask = tf.cast(tf.cast(mask, tf.bool), tf.float32)
            # set logits to -inf where mask=0 to ignore them
            # during packpropagation
            logits += (1.0 - mask) * -1e9

        # softmax is apply in axis -1 that is the dimension of keys
        attention = tf.nn.softmax(logits, axis=-1)
        output = attention @ V  # (..., lenQ, dV)
        return output, attention


    def call(self, Q: tf.Tensor, K: tf.Tensor, V: tf.Tensor,
             mask: tf.Tensor=None):
        """Multi-Head Attention block

        Parameters
        ----------
        Q : tf tensor, shape=(batch_size, lenQ, d_model)
            Queries (full depth).
        K : tf tensor, shape=(batch_size, lenK, d_model)
            Keys (full depth).
        V : tf tensor, shape=(batch_size, lenV, d_model)
            Values (full depth).
        mask : None or tf tensor, shape=(batch_size, lenQ, lenK), optional
            Mask to avoid backpropagation on some data.
            If None, no mask are used by default None.

        Raises
        ------
        ValueError:
            Q, K and V have not the same first dimension (batch_size)

        Returns
        -------
        output : tf tensor, shape=(batch_size, lenQ, d_model)
            Output of Multi-Head Attention block.
        attention : tf tensor, shape=(batch_size, n_heads, lenQ, lenK)
            Batched attention weights of the block.
        """
        if (K.get_shape()[0] != Q.get_shape()[0] or
            V.get_shape()[0] != Q.get_shape()[0]):
            raise ValueError("Q, K and V must have the same first dimension "
                             "(batch size), found respectively "
                             f"{Q.get_shape()[0]}, {K.get_shape()[0]} and "
                             f"{V.get_shape()[0]}.")

        batch_size = tf.shape(Q)[0]
        Q = self.Wq(Q) # (batch_size, lenQ, d_model)
        K = self.Wk(K)
        V = self.Wv(V)
        Q = self._split_to_heads(Q) # (batch_size, n_heads, lenQ, d_head)
        K = self._split_to_heads(K)
        V = self._split_to_heads(V)

        heads, attention = self.scaled_dot_product_attention(Q, K, V, mask)

        # invert head splitting
        heads = tf.transpose(heads, perm=[0, 2, 1, 3])
        # shape (batch_size, lenQ, n_heads, depth)
        concat_heads = tf.reshape(heads, (batch_size, -1, self.d_model))
        # shape (batch_size, lenQ, d_model)

        output = self.Wo(concat_heads) # (batch_size, lenQ, d_model)
        return output, attention
