from abc import ABCMeta, abstractmethod

import torch
import ignite


class BaseTorch(object, metaclass=ABCMeta):
    def __init__(self, config):
        self.metrics = {"Accuracy": ignite.metrics.Accuracy()}

    def set_model(self, teacher, student, optimizer):
        self.teacher = teacher
        self.student = student
        self.optimizer = optimizer

    @abstractmethod
    def teach_step(self, x, y):
        pass

    def logging_metrics(self, labels, logits):
        for name in self.metrics.keys():
            self.metrics[name].update([logits, labels])

    def reset_metrics(self):
        for name in self.metrics.keys():
            self.metrics[name].reset()

    def set_metrics(self, metrics):
        for name in metrics.keys():
            assert isinstance(metrics.keys()[name], ignite.metrics.Metric)
            self.metrics[name] = metrics

    def get_metrics(self):
        result = {}
        for name in self.metrics.keys():
            result[name] = self.metrics[name].compute()
        return result

    def test(self, dataset):
        self.reset_metrics()
        student = self.student.eval()
        with torch.set_grad_enabled(False):
            for x, y in dataset:
                logits = student(x)
                self.logging_metrics(y, logits)
        result = self.get_metrics()
        self.reset_metrics()
        return result
