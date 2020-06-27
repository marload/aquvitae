"""Ref
https://arxiv.org/pdf/1503.02531.pdf
https://github.com/sseung0703/Knowledge_distillation_via_TF2.0
"""

import tensorflow as tf
from .base_tensorflow import BaseTensorflow


class ST(BaseTensorflow):
    def __init__(self, config):
        super(ST, self).__init__(config)

        self.teacher = None
        self.student = None
        self.optimizer = None

        self.alpha = config["alpha"]
        self.T = config["T"]

    def teach_step(self, x, y):
        with tf.GradientTape() as tape:
            t_logits = self.teacher(x, training=False)
            s_logits = self.student(x, training=True)
            loss = self.compute_loss(t_logits, s_logits, y)
        grad = tape.gradient(loss, self.student.trainable_variables)
        self.optimizer.apply_gradients(zip(grad, self.student.trainable_variables))
        self.logging_metrics(y, s_logits)
        return loss

    @tf.function
    def compute_loss(self, t_logits, s_logits, labels):
        cls_loss = self.cls_loss(labels, s_logits)
        st_loss = self.st_loss(t_logits, s_logits)
        return (1 - self.alpha) * cls_loss + self.alpha * st_loss

    @tf.function
    def cls_loss(self, labels, s_logits):
        criterion = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        return criterion(labels, s_logits)

    @tf.function
    def st_loss(self, t_logits, s_logits):
        assert t_logits.shape == s_logits.shape
        return tf.reduce_mean(
            tf.reduce_sum(
                tf.nn.softmax(t_logits / self.T)
                * (
                    tf.nn.log_softmax(t_logits / self.T)
                    - tf.nn.log_softmax(s_logits / self.T)
                ),
                1,
            )
        )
