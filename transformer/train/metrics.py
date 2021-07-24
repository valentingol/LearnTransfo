import tensorflow as tf

class MaskedSparseCategoricalCrossentropy(tf.keras.losses.Loss):
    """ Computes the sparse categorical crossentropy masked
    by the labels equal to 0.
    """
    def __init__(self, from_logits: bool=False):
        """
        Parameters
        ----------
        from_logits : bool, optional
            Whether the prediction are from logits,
            by default False.
        """
        super(MaskedSparseCategoricalCrossentropy, self).__init__()
        self.from_logits = from_logits
        # reduction='none' means that base_loss returns one value per token
        self.base_loss = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=from_logits, reduction='none'
            )

    def __call__(self, y_true: tf.Tensor, y_pred: tf.Tensor):
        """Returns the loss value averaged over batch and tokens.

        Parameters
        ----------
        y_true : tf.Tensor, shape=(..., seq_len), dtype=int64
            Labels, masked value are store as 0.
        y_pred : float tf.Tensor, shape=(..., seq_len, vocab_size)
            Prediction (probability or logits) for each token.

        Returns
        -------
        loss : tf.Tensor, shape=(), dtype=float32
            Result of the sparse categorical crossentropy function
            masked by the labels equal to 0.
        """
        mask = tf.not_equal(y_true, 0)
        mask = tf.cast(mask, tf.float32)
        loss = tf.cast(self.base_loss(y_true, y_pred), tf.float32)
        loss *= mask
        # sum over batch and token
        loss = tf.reduce_sum(loss) / (tf.reduce_sum(mask) + 1e-8)
        return loss


class MaskedAccuracy(object):
    """ Computes the standard accuracy masked by the labels
    equal to 0.
    """
    def __init__(self):
        pass

    def __call__(self, y_true: tf.Tensor, y_pred: tf.Tensor):
        """Returns the accuracy value averaged over batch
        and tokens.

        Parameters
        ----------
        y_true : tf.Tensor, shape=(..., seq_len), dtype=int64
            Labels, masked value are store as 0.
        y_pred : float tf.Tensor, shape=(..., seq_len, vocab_size)
            Prediction (probability or logits) for each token.

        Returns
        -------
        acc : tf.Tensor, shape=(), dtype=float32
            Result of the accuracy masked by the labels
            equal to 0.
        """
        mask = tf.not_equal(y_true, 0)
        good_preds = tf.equal(y_true, tf.argmax(y_pred, axis=-1))
        good_preds = tf.cast(tf.logical_and(mask, good_preds), tf.float32)
        mask = tf.cast(mask, tf.float32)
        acc = tf.reduce_sum(good_preds) / (tf.reduce_sum(mask) + 1e-8)
        return acc
