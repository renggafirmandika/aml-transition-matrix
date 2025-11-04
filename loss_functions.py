import tensorflow as tf

def symmetric_cross_entropy(alpha=0.1, beta=1.0, num_classes=3, epsilon=1e-7):
    def loss_function(y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        y_true_one_hot = tf.one_hot(y_true, depth=num_classes)

        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        y_true_one_hot = tf.clip_by_value(y_true_one_hot, epsilon, 1.0)

        # standard cross-entropy
        ce = -tf.reduce_sum(y_true_one_hot * tf.math.log(y_pred), axis=-1)

        # reverse cross-entropy
        rce = -tf.reduce_sum(y_pred * tf.math.log(y_true_one_hot), axis=-1)

        # symmetric cross-entropy
        loss = alpha * ce + beta * rce
        
        return tf.reduce_mean(loss)

    return loss_function


def forward_correction_loss(transition_matrix, num_classes = 3, epsilon = 1e-7):
    T = tf.constant(transition_matrix, dtype = tf.float32)

    def loss_function(y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        y_true_one_hot = tf.one_hot(y_true, depth = num_classes)

        y_pred_clipped = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        #Works for a batch of vectors
        p_tilde = tf.matmul(y_pred_clipped, T)
        #Numerical stability
        p_tilde_clipped = tf.clip_by_value(p_tilde, epsilon, 1.0 - epsilon)

        ce_loss = -tf.reduce_sum(y_true_one_hot * tf.math.log(p_tilde_clipped), axis = -1)

        return tf.reduce_mean(ce_loss)
    return loss_function



