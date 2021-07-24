import tensorflow as tf
kl = tf.keras.layers

class PostionalEncoding(kl.Layer):
    """Computes the positional encoding with sinus/cosinus found in
    the original transformer implementation (Vaswani et al, 2017).
    """
    def __init__(self, seq_len: int, depth: int, max_denom: int=10000):
        """
        Parameters
        ----------
        seq_len : int
            Sequence length (should be at least the maximum
            sequence length of data).
        max_depth : int
            Depth of model.
        max_denom : int, optional
            Maximum positional denominator in angles. The maximum
            wavelength (regarding positions) of the positinal encoding
            is equal to 2pi*max_denom.
        """
        super(PostionalEncoding, self).__init__()
        self.seq_len = seq_len
        self.depth = depth
        self.max_denom = max_denom

    def _alternate_concat(self, X: tf.Tensor, Y: tf.Tensor):
        """Concate input tensor along last dimension by alternating the
        values. The tensors must have the same shape.

        Parameters
        ----------
        X : tf.Tensor, size=(..., d)
            Fist tensor to concatenate.
        Y : tf.Tensor, size=(..., d)
            Second tensor to concatenate.

        Raises
        ------
        ValueError :
            If the tensors have different shapes.

        Returns
        -------
        concat : tf.Tensor, size=(..., 2*d)
            Concatenated tensor by alterning values of X and Y
            along the last dimension.
        """
        if X.get_shape() != Y.get_shape():
            raise ValueError("X and Y should have the same shape, found"
                             f"{X.get_shape()} and {Y.get_shape()}.")
        X = tf.expand_dims(X, axis=-1)
        Y = tf.expand_dims(Y, axis=-1)
        concat = tf.concat([X, Y], axis=-1)
        shape = [tf.shape(X)[i] for i in range(len(tf.shape(X)) - 2)]
        shape.append(-1)
        concat = tf.reshape(concat, shape)
        return concat


    def __call__(self):
        """Returns a tensor of shape (seq_len, depth) that represents
        the positional encoding.

        Returns
        -------
        pos_code : tf.Tensor, shape=(seq_len, depth)
            Result of the positional encoding.
        """
        seq_len, depth = self.seq_len, self.depth
        pos_range = tf.range(seq_len)
        dep_range = tf.range(depth)
        positions, depths = tf.meshgrid(pos_range, dep_range, indexing='ij')
        # size (seq_len, depth), (seq_len, depth)
        positions = tf.cast(positions, tf.float32)
        # size (seq_len, depth)
        powers = tf.cast(2 * depths//2, tf.float32) / depth
        denom = tf.cast(self.max_denom, tf.float32)
        angles = positions / tf.pow(denom, powers)
        code_sin = tf.sin(angles[:, 0::2])
        code_cos = tf.cos(angles[:, 1::2])
        # if depth is odd, add a virtual line at the end of
        # encode_cos to have the same shape as encode_sin
        if depth % 2 != 0:
            code_cos = tf.concat([code_cos, tf.zeros((seq_len, 1))],
                                   axis=-1)
            pos_code = self._alternate_concat(code_sin, code_cos)
            pos_code = pos_code[:, :-1]
        else:
            pos_code = self._alternate_concat(code_sin, code_cos)
        return pos_code
