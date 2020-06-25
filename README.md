> This repository is still in development.

![TF Depend](https://img.shields.io/badge/TensorFlow-2.1-orange) ![TORCH Depend](https://img.shields.io/badge/pytorch-1.5.1-blue) ![License Badge](https://img.shields.io/badge/license-MIT-green)<br>

<p align="center">
  <img width="500" src="./assets/logo.png">
</p>

<h2 align=center>The easiest Knowledge Distillation Library</h2>

## Getting Started

```python
from tensorflow as tf
from quavitae import dist, KD

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

train_ds = tf.data.Dataset.from_tensor_slices(
    (x_train, y_train)).shuffle(10000).batch(64)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

# NOTE: Load the teacher and student model
teacher = ...
student = ...


dist(
    teacher=teacher,
    student=student,
    algo=KD(alpha=0.6, T=2.5),
    optimizer=tf.keras.optimizers.Adam(),
    train_ds=train_ds,
    test_ds=test_ds
)
```