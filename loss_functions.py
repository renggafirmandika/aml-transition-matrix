import tensorflow as tf

def symmetric_cross_entropy(alpha=0.1, beta=1.0, A=-1.0, num_classes=3, epsilon=1e-7):
    def loss_function(y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        y_true_one_hot = tf.one_hot(y_true, depth=num_classes)

        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        y_true_one_hot = tf.clip_by_value(y_true_one_hot, epsilon, 1.0)

        # standard cross-entropy
        ce = -tf.reduce_sum(y_true_one_hot * tf.math.log(y_pred), axis=-1)

        # reverse cross-entropy
        pred_correct = tf.reduce_sum(y_true_one_hot * y_pred, axis=-1)
        pred_correct = tf.clip_by_value(pred_correct, epsilon, 1.0)
        rce = -A*(1.0-pred_correct)

        # symmetric cross-entropy
        loss = alpha * ce + beta * rce
        
        return tf.reduce_mean(loss)

    return loss_function

