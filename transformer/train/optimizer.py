import tensorflow as tf

class ScheduleLR(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self, d_model: int, warmup_steps: int=4000,
                 min_lr: float=1e-4):
        """
        Parameters
        ----------
        d_model : int
            Depth of the model.
        warmup_steps : int, optional
            Number of steps where the learning rate warm up,
            by default 4000.
        min_lr : float, optional
            Minimum learning rate, by default 1e-4.
        """
        super(ScheduleLR, self).__init__()
        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = tf.cast(warmup_steps, tf.float32)
        self.min_lr = tf.cast(min_lr, tf.float32)

    def __call__(self, step:int):
        """Returns the learning rate at step 'step'."""
        step = tf.cast(step, tf.float32)
        lr1 = tf.math.rsqrt(step)
        lr2 = step * (self.warmup_steps ** -1.5)
        lr3 = self.min_lr
        lr = tf.maximum(tf.minimum(lr1, lr2) * tf.math.rsqrt(self.d_model),
                        lr3)
        return lr


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    lr_schedule = ScheduleLR(d_model=512)
    plt.plot(lr_schedule(tf.range(40000, dtype=tf.float32)))
    plt.show()
