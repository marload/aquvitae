import tensorflow as tf
from aquvitae import dist, DML

dataset = tf.keras.datasets.cifar100
(x_train, y_train), (x_test, y_test) = dataset.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

train_ds = (
    tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(128)
)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(128)

teacher = tf.keras.applications.ResNet50(
    include_top=True,
    weights=None,
    input_tensor=None,
    input_shape=(32, 32, 3),
    pooling=None,
    classes=100,
    classifier_activation=None,
)
student = tf.keras.applications.ResNet50(
    include_top=True,
    weights=None,
    input_tensor=None,
    input_shape=(32, 32, 3),
    pooling=None,
    classes=100,
    classifier_activation=None,
)
teacher.load_weights("./tutorials/best_model.h5")

optimizer = tf.keras.optimizers.Adam()
st = DML(alpha=1.0)

student = dist(
    teacher=teacher,
    student=student,
    algo=st,
    optimizer=tf.keras.optimizers.Adam(),
    train_ds=train_ds,
    test_ds=test_ds,
    iterations=50000,
)
