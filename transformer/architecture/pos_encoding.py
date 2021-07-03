import tensorflow as tf
kl = tf.keras.layers

class PostionalEncoding(kl.Layer):
    """Computes the positional encoding with sinus/cosinus found in
    the original transformer implementation (Vaswani et al, 2017).
    """
    def __init__(self, max_denom: int=10000):
        """
        Parameters
        ----------
        max_denom : int, optional
            Maximum positional denominator in angles. The maximum
            wavelength (regarding positions) of the positinal encoding
            is equal to 2pi*max_denom.
        """
        self.max_denom = max_denom

    def alternate_concat(self, X: tf.Tensor, Y: tf.Tensor):
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
        ValueEror :
            X and Y do not have the same shape.

        Returns
        -------
        concat : tf.Tensor, size=(..., 2*d)
            Concatenated tensor by alterning values of X and Y
            along the last dimension.
        """
        if X.shape != Y.shape:
            raise ValueError("X and Y do not have the same shape (found "
                             f"{X.shape} and {Y.shape})")
        X = tf.expand_dims(X, axis=-1)
        Y = tf.expand_dims(Y, axis=-1)
        concat = tf.concat([X, Y], axis=-1)
        concat = tf.reshape(concat, (*X.shape[:-2], -1))
        return concat


    def __call__(self, seq_len: int, depth: int):
        """Returns a tensor of shape (seq_len, depth) that represents
        the positional encoding.

        Parameters
        ----------
        seq_len : int
            Sequence length.
        max_depth : int
            Depth.

        Returns
        -------
        pos_code : tf.Tensor, shape=(seq_len, depth)
            Result of the positional encoding.
        """
        pos_range = tf.range(seq_len)
        dep_range = tf.range(depth)
        positions, depths = tf.meshgrid(pos_range, dep_range, indexing='ij')
        # size (seq_len, depth), (seq_len, depth)
        positions = tf.cast(positions, tf.float32)
        # size (seq_len, depth)
        powers = tf.cast(2 * depths//2, tf.float32) / depth
        angles = positions / tf.pow(self.max_denom, powers)
        code_sin = tf.sin(angles[:, 0::2])
        code_cos = tf.cos(angles[:, 1::2])
        # if depth is odd, add a virtual line at the end of
        # encode_cos to have the same shape as encode_sin
        if depth % 2 != 0:
            code_cos = tf.concat([code_cos, tf.zeros((seq_len, 1))],
                                   axis=-1)
            pos_code = self.alternate_concat(code_sin, code_cos)
            pos_code = pos_code[:, :-1]
        else:
            pos_code = self.alternate_concat(code_sin, code_cos)
        return pos_code

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    max_denom = 1000
    seq_len = 7000
    depth = 30
    pe = PostionalEncoding(max_denom=max_denom)
    pos_code = pe(seq_len=seq_len, depth=depth)
    code_sin = pos_code[:, 0::2]
    code_cos = pos_code[:, 1::2]

    plt.subplot(211)
    plt.title('sinus encoding')
    plt.pcolormesh(code_sin.numpy().T, cmap='coolwarm')
    plt.ylabel('depths')
    plt.xlabel('positions')
    plt.colorbar()

    plt.subplot(212)
    plt.title('cosinus encoding')
    plt.pcolormesh(code_cos.numpy().T, cmap='coolwarm')
    plt.ylabel('depths')
    plt.xlabel('positions')
    plt.colorbar()

    plt.show()
