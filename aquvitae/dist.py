import os
from tqdm import tqdm

import torch
import tensorflow as tf


from .algo import BaseKD
from .utils import result_to_tqdm_template


def dist(
    teacher, student, algo, optimizer, train_ds, test_ds, iterations, test_freq=200
):
    info = _check_dist(teacher, student, algo, optimizer, train_ds, test_ds,)

    if info["framework"] == "torch":
        return _torch_dist(
            teacher,
            student,
            algo.torch(),
            optimizer,
            train_ds,
            test_ds,
            iterations,
            test_freq,
        )
    elif info["framework"] == "tensorflow":
        return _tensorflow_dist(
            teacher,
            student,
            algo.tensorflow(),
            optimizer,
            train_ds,
            test_ds,
            iterations,
            test_freq,
        )
    else:
        raise NotImplementedError


def _check_dist(teacher, student, algo, optimizer, train_ds, test_ds):
    assert isinstance(algo, BaseKD)

    info = {"framework": None}
    if isinstance(teacher, torch.nn.Module) and isinstance(student, torch.nn.Module):
        info["framework"] = "torch"
    elif isinstance(teacher, tf.keras.Model) and isinstance(student, tf.keras.Model):
        assert isinstance(optimizer, tf.keras.optimizers.Optimizer)
        assert isinstance(train_ds, tf.data.Dataset)
        assert isinstance(test_ds, tf.data.Dataset)
        info["framework"] = "tensorflow"

    return info


def _tensorflow_dist(
    teacher, student, algo, optimizer, train_ds, test_ds, iterations, test_freq
):
    algo.set_model(teacher, student, optimizer)
    train_ds, test_ds = iter(train_ds), iter(test_ds)

    iter_log = tqdm(range(iterations), position=0)
    train_metric_log = tqdm(total=0, desc="TRAIN", position=1, bar_format="{desc}")
    test_metric_log = tqdm(total=0, desc="TEST", position=2, bar_format="{desc}")

    for it in iter_log:
        x, y = next(train_ds)
        loss = algo.teach_step(x, y)
        result = algo.get_metrics()
        train_metric_tmp = result_to_tqdm_template(result)
        train_metric_log.set_description_str("TRAIN\t" + train_metric_tmp)
        iter_log.update(1)

        if it % test_freq == 0 and it != 0:
            algo.reset_metrics()
            result = algo.test(test_ds)
            test_metric_tmp = result_to_tqdm_template(result)
            test_metric_log.set_description_str("TEST\t" + test_metric_tmp)

    return student


def _torch_dist(teacher, student, algo, optimizer, dataset, iterations, test_freq):
    raise NotImplementedError
