import tensorflow as tf
from tensorflow import keras

# define the loss function of SCE
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

# forward correction
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

# coteaching
class CoTeachingProxyLoss(keras.losses.Loss):
    def __init__(self, remember_rate=0.7, num_classes=3, epsilon=1e-7, name="co_teaching_proxy"):
        super().__init__(name=name)
        self.remember_rate = tf.Variable(float(remember_rate), trainable=False, dtype=tf.float32)
        self.num_classes = int(num_classes)
        self.epsilon = float(epsilon)

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        y_true_oh = tf.one_hot(y_true, depth=self.num_classes)

        # If model outputs logits, uncomment:
        # y_pred = tf.nn.softmax(y_pred, axis=-1)

        y_pred = tf.clip_by_value(y_pred, self.epsilon, 1.0 - self.epsilon)
        ce = -tf.reduce_sum(y_true_oh * tf.math.log(y_pred), axis=-1)  # per-example CE [B]

        # Select k = ceil(B * remember_rate) smallest losses
        bsz = tf.shape(ce)[0]
        k = tf.cast(tf.math.maximum(1.0, tf.cast(bsz, tf.float32) * self.remember_rate), tf.int32)
        _, idx = tf.math.top_k(-ce, k=k, sorted=False)  # negative => smallest
        selected = tf.gather(ce, idx)
        return tf.reduce_mean(selected)

class RememberRateScheduler(keras.callbacks.Callback):
    def __init__(self, loss_obj, max_epochs, noise_rate=0.4, warmup=5):
        super().__init__()
        self.loss_obj = loss_obj
        self.max_epochs = int(max_epochs)
        self.noise_rate = float(noise_rate)
        self.warmup = int(warmup)

    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.warmup:
            rr = 1.0
        else:
            t = (epoch - self.warmup) / max(1, (self.max_epochs - self.warmup))
            t = float(min(max(t, 0.0), 1.0))
            rr = 1.0 - self.noise_rate * t
        self.loss_obj.remember_rate.assign(rr)

def _infer_noise_rate_from_name(dataset_name: str):
    s = str(dataset_name).lower()
    if "0.6" in s: return 0.6
    if "0.3" in s: return 0.3
    return 0.4  
