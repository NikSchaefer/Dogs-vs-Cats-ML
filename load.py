import tensorflow as tf


SAVE_PATH = "C:\\Users\\schaefer\\Desktop\\ML\\Dogs-vs-Cats-ML\\save\\model.ckpt"

model = tf.keras.models.load_model(SAVE_PATH)

print(model.summary())
