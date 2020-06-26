import tensorflow as tf

dataset = tf.keras.datasets.cifar100
(x_train, y_train), (x_test, y_test) = dataset.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

crietrion = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

check_point = tf.keras.callbacks.ModelCheckpoint(
    "best_model.h5", monitor="val_accuracy", mode="max", save_best_only=True
)


model = tf.keras.applications.ResNet50(
    include_top=True,
    weights=None,
    input_tensor=None,
    input_shape=(32, 32, 3),
    pooling=None,
    classes=100,
    classifier_activation=None,
)

model.compile(optimizer="adam", loss=crietrion, metrics=["accuracy"])

model.fit(
    x_train,
    y_train,
    epochs=100,
    batch_size=128,
    validation_data=(x_test, y_test),
    callbacks=[check_point],
)
