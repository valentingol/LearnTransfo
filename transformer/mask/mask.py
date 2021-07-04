import tensorflow as tf

def get_padding_mask(token: tf.Tensor):
    """Returns padding mask considering 0 as the pad token.
    The result is broadcastable with logits in attention
    operation.

    Parameters
    ----------
    token : tf.Tensor of integers, shape=(..., seq_len)
        Token tensor with zero padding.

    Returns
    -------
    padding_mask : tf.Tensor, shape=(..., 1, 1, seq_len)
        Padding mask corresponding to token_seq.
        0 for pad values, 1 for other values.
        The additional dimensions allow the mask to be
        broadcastable with the heads and the queries dimension
        of logits in the attention operation.
    """
    padding_mask = tf.cast(tf.minimum(token, 1), tf.int32)
    # add extra dimensions to add the padding
    # to the attention logits.
    padding_mask = padding_mask[..., tf.newaxis, tf.newaxis, :]
    return padding_mask


def get_future_mask(token: tf.Tensor):
    """Returns the future mask that can be used for preventing the
    decoder to look future values. The result is broadcastable with
    logits in attention operation.

    Parameters
    ----------
    token : tf.Tensor of integers, shape=(..., seq_len)
        Token tensor. Only the size of last dimension is used
        to create the mask.

    Returns
    -------
    future_mask : tf.Tensor, shape=(1, 1, seq_len, seq_len)
        Lower-triangular matrix that corresponds to the mask of
        the future values.
        0 for rejected values, 1 for other values.
        The additional dimensions allow the mask to be
        broadcastable with the batch and the heads dimension of
        logits in the attention operation.
    """
    seq_len = token.shape[-1]
    ones = tf.ones((seq_len, seq_len), dtype=tf.int32)
    future_mask = tf.linalg.band_part(ones, -1, 0)
    future_mask = future_mask[tf.newaxis, tf.newaxis, ...]
    return future_mask


def get_masks(inputs_token_seq: tf.Tensor, outputs_token_seq: tf.Tensor):
    """Returns input padding mask and combined output mask
    (product of output padding mask and future mask). The results
    are broadcastable with logits in attention operation.

    Parameters
    ----------
    inputs_token_seq : tf.Tensor
        Input token sequences with zero padding.
    outputs_token_seq : tf.Tensor
        Output token sequences with zero padding.

    Returns
    -------
    in_padding_mask : tf.Tensor, shape=(..., 1, 1, seq_len)
        Padding mask corresponding to input sequence.
        0 for pad values, 1 for other values.
        The additional dimensions allow the mask to be
        broadcastable with the heads and the queries dimension
        of logits in the attention operation.
    out_mask : tf tensor, shape=(..., 1, seq_len, seq_len)
        Mask corresponding to output sequence. It combines the
        padding mask of output data and the future mask.
        0 for rejected values, 1 for other values.
        The additional dimension allows the mask to be
        broadcastable with the heads dimension of logits in
        the attention operation.
    """
    in_pad_mask = get_padding_mask(inputs_token_seq)
    out_pad_mask = get_padding_mask(outputs_token_seq)
    future_mask = get_future_mask(outputs_token_seq)
    # out_pad_mask and future_mask are broadcasted together
    out_mask = future_mask * out_pad_mask
    return in_pad_mask, out_mask
