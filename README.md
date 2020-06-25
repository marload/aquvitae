> This repository is still in development.

![TF Depend](https://img.shields.io/badge/TensorFlow-2.1-orange) ![TORCH Depend](https://img.shields.io/badge/pytorch-1.5.1-blue) ![License Badge](https://img.shields.io/badge/license-MIT-green)<br>

<p align="center">
  <img width="400" src="./assets/logo.png">
</p>

<h2 align=center>The easiest Knowledge Distillation Library</h2>

## Installation

```base
# pip install aquvitae
```

## Getting Started

### TensorFlow Example
```python
from tensorflow as tf
from quavitae import dist, ST

# Load the dataset
train_ds = ...
test_ds = ...

# Load the teacher and student model
teacher = ...
student = ...


student = dist(
    teacher=teacher,
    student=student,
    algo=ST(alpha=0.6, T=2.5),
    optimizer=tf.keras.optimizers.Adam(),
    train_ds=train_ds,
    test_ds=test_ds
)
```

### PyTorch Example
```python
from torch
from quavitae import dist, DML

# Load the dataset
train_ds = ...
test_ds = ...

# Load the teacher and student model
teacher = ...
student = ...


student = dist(
    teacher=teacher,
    student=student,
    algo=DML(alpha=0.6),
    optimizer=torch.optim.Adam()
    train_ds=train_ds,
    test_ds=test_ds
)
```