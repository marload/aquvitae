import tensorflow as tf
from .base_tensorflow import BaseTensorflow


class DML(BaseTensorflow):
    def __init__(self, config):
        super(DML, self).__init__(config)

        self.teacher = None
        self.student = None
        self.optimizer = None

        self.alpha = config['alpha']

    def teach_step(self, x, y):
        with tf.GradientTape() as tape:
            t_logits = self.teacher(x, training=False)
            s_logits = self.student(x, training=True)
            loss = self.compute_loss(t_logits, s_logits, y)
        grad = tape.gradient(loss, self.student.trainable_variables)
        self.optimizer.apply_gradients(
            zip(grad, self.student.trainable_variables))
        self.logging_metrics(y, s_logits)
        return loss

    def compute_loss(self, t_logits, s_logits, labels):
        return (1-self.alpha) * self.cls_loss(labels, s_logits) + self.alpha * self.st_loss(t_logits, s_logits)

    def cls_loss(self, labels, s_logits):
        s_logits = tf.nn.softmax(s_logits)
        criterion = tf.keras.losses.SparseCategoricalCrossentropy()
        return criterion(labels, s_logits)

    def st_loss(self, t_logits, s_logits):
        assert t_logits.shape == s_logits.shape
        return (tf.reduce_mean(tf.reduce_sum(tf.nn.softmax(t_logits)*(tf.nn.log_softmax(t_logits)-tf.nn.log_softmax(s_logits)), 1)) +
                tf.reduce_mean(tf.reduce_sum(tf.nn.softmax(s_logits)*(tf.nn.log_softmax(s_logits)-tf.nn.log_softmax(t_logits)), 1)))/2
