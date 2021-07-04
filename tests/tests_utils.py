import tensorflow as tf

def tf_equal(A: tf.Tensor, B:tf.Tensor):
    """Returns whether A and B have all their components closed
    (returns True), or not (returns False). A and B must be
    broadcastable together.

    Parameters
    ----------
    A : tf.Tensor
    B : tf.Tensor

    Returns
    -------
    bool
    """
    return tf.math.reduce_all(tf.experimental.numpy.isclose(A, B)).numpy()