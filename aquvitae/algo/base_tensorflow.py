from tqdm import tqdm
from abc import ABCMeta, abstractmethod

import tensorflow as tf


class BaseTensorflow(object, metaclass=ABCMeta):
    def __init__(self, config):
        self.metrics = [tf.keras.metrics.SparseCategoricalAccuracy(name="Accuracy")]

    def set_model(self, teacher, student, optimizer):
        self.teacher = teacher
        self.student = student
        self.optimizer = optimizer

    @abstractmethod
    def teach_step(self, x, y):
        pass

    def logging_metrics(self, labels, logits):
        for metric in self.metrics:
            metric.update_state(labels, logits)

    def reset_metrics(self):
        for metric in self.metrics:
            metric.reset_states()

    def set_metrics(self, metrics):
        for metric in metrics:
            assert isinstance(metric, tf.keras.metrics.Metric)
            self.metrics.append(metrics)

    def get_metrics(self):
        result = {}
        for metric in self.metrics:
            result[metric.name] = metric.result()
        return result

    def test(self, dataset):
        self.reset_metrics()
        for x, y in dataset:
            logits = self.student(x, training=False)
            self.logging_metrics(y, logits)
        result = self.get_metrics()
        self.reset_metrics()
        return result
