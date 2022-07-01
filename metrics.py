import tensorflow as tf

def masked_reduce_mean(err, gt_depth):
    mask = tf.cast(tf.greater(gt_depth, 1e-6), tf.float32)
    return tf.reduce_sum(tf.math.multiply_no_nan(err,mask))/tf.maximum(tf.reduce_sum(mask),1)


class RootMeanSquaredError(tf.keras.metrics.Mean):

    def __init__(self, name='RMSE', **kwargs):
        super(RootMeanSquaredError, self).__init__(name=name, **kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):
        error = tf.square(y_true-y_pred)
        error = tf.sqrt(masked_reduce_mean(error, y_true))
        return super(RootMeanSquaredError, self).update_state(error)

class RootMeanSquaredLogError(tf.keras.metrics.Mean):

    def __init__(self, name='RMSE_log', **kwargs):
        super(RootMeanSquaredLogError, self).__init__(name=name, **kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):
        mask = tf.cast(tf.greater(y_true, 0.), tf.float32)
        y_true = tf.math.log(y_true+1e-6)
        y_pred = tf.math.log(y_pred+1e-6)
        error = tf.square(y_true-y_pred)
        error = tf.sqrt(masked_reduce_mean(error, y_true))

        return super(RootMeanSquaredLogError, self).update_state(error)

class AbsRelError(tf.keras.metrics.Mean):

    def __init__(self, name='AbsRel', **kwargs):
        super(AbsRelError, self).__init__(name=name, **kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):
        error = tf.math.abs(y_true - y_pred) / (y_true+1e-6)
        error = masked_reduce_mean(error, y_true)
        return super(AbsRelError, self).update_state(error)


class SqRelError(tf.keras.metrics.Mean):

    def __init__(self, name='SqRel', **kwargs):
        super(SqRelError, self).__init__(name=name, **kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):
        error = tf.math.squared_difference(y_true, y_pred) / (y_true+1e-6)
        error = masked_reduce_mean(error, y_true)
        return super(SqRelError, self).update_state(error)


class ThresholdRelError(tf.keras.metrics.Mean):

    def __init__(self, threshold, name='Delta', **kwargs):
        self.threshold = threshold
        super(ThresholdRelError, self).__init__(name=name + str(threshold), **kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):
        thresh = tf.maximum((y_true / y_pred), (y_pred / y_true))
        error = tf.cast(tf.math.less(thresh, 1.25 ** self.threshold), tf.float32)
        error = masked_reduce_mean(error, y_true)
        return super(ThresholdRelError, self).update_state(error)
