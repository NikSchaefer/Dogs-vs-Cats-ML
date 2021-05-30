import numpy as np
import tensorflow as tf
from keras.preprocessing.image import load_img
import matplotlib.pyplot as plt
IMG_SIZE = 224

SAVE_PATH = "C:\\Users\\schaefer\\Desktop\\ML\\Dogs-vs-Cats-ML\\save\\model.ckpt"

img = load_img("data/test/test/5.jpg", target_size=(IMG_SIZE, IMG_SIZE)) # dog or cat image
img = np.asarray(img)
plt.imshow(img)
plt.show()
img = np.expand_dims(img, axis=0)

model = tf.keras.models.load_model(SAVE_PATH)

output = model.predict(img)

print(output)
if output[0][0] > output[0][1]:
    print("cat")
else:
    print('dog')
