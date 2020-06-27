import tensorflow as tf
from aquvitae import dist, ST

TEACHER_WEIGHTS_PATH = ...  # Teacher Weights PATH
BATCH_SIZE = 64
INPUT_SHAPE = (32, 32, 3)
NUM_CLASSES = 10

# Load the dataset
cifar10 = tf.keras.datasets.cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

train_ds = (
    tf.data.Dataset.from_tensor_slices((x_train, y_train))
    .shuffle(10000)
    .batch(BATCH_SIZE)
)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(BATCH_SIZE)

# Load the teacher and student model
teacher = tf.keras.applications.ResNet50(
    weights=None,
    input_shape=INPUT_SHAPE,
    classes=NUM_CLASSES,
    # teacher's output activation function must be `None`.
    classifier_activation=None,
)
student = tf.keras.applications.MobileNetV2(
    weights=None,
    input_shape=INPUT_SHAPE,
    classes=NUM_CLASSES,
    # student's output activation function must be `None`.
    classifier_activation=None,
)
teacher.load_weights(TEACHER_WEIGHTS_PATH)

student = dist(
    teacher=teacher,
    student=student,
    algo=ST(alpha=0.6, T=2.5),  # Knowledge Distillation Algorithm Instance
    optimizer=tf.keras.optimizers.Adam(),
    train_ds=train_ds,  # `train_ds` must be an Instance of `tf.data.Dataset`.
    test_ds=test_ds,  # `test_ds` must be an Instance of `tf.data.Dataset`.
    iterations=3000,
)
