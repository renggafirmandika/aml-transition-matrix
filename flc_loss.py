# flc_loss.py
import tensorflow as tf

def forward_correction_loss(T, num_classes, epsilon=1e-7):
    """
    Forward Loss Correction
      p_noisy = p_clean @ T
      loss = CE(y_true, p_noisy)
    """
    T_tf = tf.constant(T, dtype=tf.float32)
    C = int(num_classes)

    def loss_fn(y_true, y_pred_clean):

        y_true = tf.cast(y_true, tf.int32)
        y_one  = tf.one_hot(y_true, depth=C)
        y_pred_clean = tf.clip_by_value(y_pred_clean, epsilon, 1.0 - epsilon)
        p_noisy = tf.clip_by_value(tf.matmul(y_pred_clean, T_tf), epsilon, 1.0 - epsilon)

        ce = tf.keras.losses.categorical_crossentropy(y_one, p_noisy)
        return tf.reduce_mean(ce)
    return loss_fn