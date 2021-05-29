import tensorflow as tf


model = tf.keras.models.load_model("/export")

print(model.summary())
