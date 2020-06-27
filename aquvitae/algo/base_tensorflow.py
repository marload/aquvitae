from abc import ABCMeta, abstractmethod

import tensorflow as tf


class BaseTensorflow(object, metaclass=ABCMeta):
    def __init__(self, config):
        self.metrics = {"accuracy": tf.keras.metrics.SparseCategoricalAccuracy()}

    def set_model(self, teacher, student, optimizer):
        self.teacher = teacher
        self.student = student
        self.optimizer = optimizer

    @abstractmethod
    def teach_step(self, x, y):
        pass

    def logging_metrics(self, labels, logits):
        for name in self.metrics.keys():
            self.metrics[name].update_state(labels, logits)

    def reset_metrics(self):
        for name in self.metrics.keys():
            self.metrics[name].reset_states()

    def set_metrics(self, metrics):
        for name in metrics.keys():
            assert isinstance(metrics.keys()[name], tf.keras.metrics.Metric)
            self.metrics[name] = metrics

    def get_metrics(self):
        result = {}
        for name in self.metrics.keys():
            result[name] = self.metrics[name].result()
        return result

    @tf.function
    def test(self, dataset):
        self.reset_metrics()
        for x, y in dataset:
            logits = self.student(x, training=True)
            self.logging_metrics(y, logits)

        result = self.get_metrics()
        self.reset_metrics()
        return result
