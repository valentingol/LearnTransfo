import tensorflow as tf
kl = tf.keras.layers

class FeedForward(kl.Layer):

    def __init__(self, d_model: int, d_ff: int, activation='gelu'):
        """
        Parameters
        ----------
        d_model : int
            Depth of the model.
        dff : int
            Hidden layer size.
        activation : callable or string, optional
            Activation function of the hidden layer. Must be a valid
            activation argument in keras layers, by default 'relu'.
        """
        super(FeedForward, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.activation = activation
        self.model = tf.keras.Sequential([
            kl.Dense(self.d_ff, activation=self.activation),
            kl.Dense(self.d_model, activation='linear')
            ])

    def __call__(self, X):
        output = self.model(X)
        return output
