import os
from tqdm import trange

import torch
import tensorflow as tf


from .algo import BaseKD
from .utils import result_to_tqdm_template


def dist(
    teacher,
    student,
    algo,
    optimizer,
    train_ds,
    test_ds,
    iterations,
    metrics={},
    test_freq=200,
):
    info = _check_dist(teacher, student, algo, optimizer, train_ds, test_ds)

    if info["framework"] == "torch":
        kd = algo.torch()
        kd.set_metrics(metrics)
        return _torch_dist(
            teacher.eval(),
            student,
            kd,
            optimizer,
            train_ds,
            test_ds,
            iterations,
            test_freq,
        )
    elif info["framework"] == "tensorflow":
        kd = algo.tensorflow()
        kd.set_metrics(metrics)
        return _tensorflow_dist(
            teacher, student, kd, optimizer, train_ds, test_ds, iterations, test_freq,
        )
    else:
        raise NotImplementedError


def _check_dist(teacher, student, algo, optimizer, train_ds, test_ds):
    assert isinstance(algo, BaseKD)

    info = {"framework": None}
    if isinstance(teacher, torch.nn.Module) and isinstance(student, torch.nn.Module):
        assert isinstance(optimizer, torch.optim.Optimizer)
        assert isinstance(train_ds, torch.utils.data.DataLoader)
        assert isinstance(test_ds, torch.utils.data.DataLoader)
        info["framework"] = "torch"
    elif isinstance(teacher, tf.keras.Model) and isinstance(student, tf.keras.Model):
        assert isinstance(optimizer, tf.keras.optimizers.Optimizer)
        assert isinstance(train_ds, tf.data.Dataset)
        assert isinstance(test_ds, tf.data.Dataset)
        info["framework"] = "tensorflow"

    return info


def _init_dist(teacher, student, algo, optimizer, iterations):
    algo.set_model(teacher, student, optimizer)
    bar_format = "{desc} - {n_fmt}/{total_fmt} [{bar:30}] ELP: {elapsed}{postfix}"
    process_log = trange(iterations, desc="Training", position=0, bar_format=bar_format)
    return algo, process_log


def _tensorflow_dist(
    teacher, student, algo, optimizer, train_ds, test_ds, iterations, test_freq
):
    algo, process_log = _init_dist(teacher, student, algo, optimizer, iterations)

    train_tmp = ""
    test_tmp = ""

    train_ds = train_ds.repeat()
    for idx, (x, y) in enumerate(train_ds):
        if idx >= iterations:
            break
        process_log.update(1)
        loss = algo.teach_step(x, y)
        result = algo.get_metrics()
        train_tmp = result_to_tqdm_template(result)

        if idx % test_freq == 0 and idx != 0:
            process_log.set_description_str("Testing ")
            algo.reset_metrics()
            result = algo.test(test_ds)
            test_tmp = result_to_tqdm_template(result, training=False)
            process_log.set_description_str("Training")
        postfix = train_tmp + "- " + test_tmp
        process_log.set_postfix_str(postfix)

    return student


def _torch_dist(
    teacher, student, algo, optimizer, train_ds, test_ds, iterations, test_freq
):
    algo, process_log = _init_dist(teacher, student, algo, optimizer, iterations)

    train_tmp = ""
    test_tmp = ""

    train_iter = iter(train_ds)
    for idx in range(iterations):
        try:
            x, y = train_iter.next()
        except StopIteration:
            train_iter = iter(train_ds)
            x, y = train_iter.next()
        process_log.update(1)
        loss = algo.teach_step(x, y)
        result = algo.get_metrics()
        train_tmp = result_to_tqdm_template(result)

        if idx % test_freq == 0 and idx != 0:
            process_log.set_description_str("Testing ")
            algo.reset_metrics()
            result = algo.test(test_ds)
            test_tmp = result_to_tqdm_template(result, training=False)
            process_log.set_description_str("Training")
        postfix = train_tmp + "- " + test_tmp
        process_log.set_postfix_str(postfix)

    return student
