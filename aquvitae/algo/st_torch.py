"""Ref
https://github.com/AberHu/Knowledge-Distillation-Zoo
https://github.com/sseung0703/Knowledge_distillation_via_TF2.0
"""


import torch
import torch.nn.functional as F
from .base_torch import BaseTorch


class ST(BaseTorch):
    def __init__(self, config):
        super(ST, self).__init__(config)

        self.teacher = None
        self.student = None
        self.optimizer = None

        self.alpha = config["alpha"]
        self.T = config["T"]

    def teach_step(self, x, y):
        x = x.to(self.device)
        y = y.to(self.device)
        self.optimizer.zero_grad()
        with torch.set_grad_enabled(False):
            t_logits = self.teacher(x)
        s_logits = self.student(x)
        self.logging_metrics(y, s_logits)
        loss = self.compute_loss(t_logits, s_logits, y)
        loss.backward()
        self.optimizer.step()
        return loss

    def compute_loss(self, t_logits, s_logits, labels):
        t_logits = t_logits.detach()
        return (1 - self.alpha) * self.cls_loss(
            labels, s_logits
        ) + self.alpha * self.st_loss(t_logits, s_logits)

    def cls_loss(self, labels, s_logits):
        criterion = torch.nn.CrossEntropyLoss()
        return criterion(s_logits, labels)

    def st_loss(self, t_logits, s_logits):
        assert t_logits.shape == s_logits.shape
        return (
            F.kl_div(
                F.log_softmax(s_logits / self.T, dim=1),
                F.softmax(t_logits / self.T, dim=1),
                size_average=False,
            )
            * (self.T ** 2)
            / s_logits.shape[0]
        )
