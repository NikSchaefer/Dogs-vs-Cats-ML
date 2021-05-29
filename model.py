import pandas as pd
import numpy as np
from tensorflow.python.keras.applications.vgg16 import VGG16
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Input
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from tensorflow.python.keras.engine.input_layer import Input
from keras.optimizers import Adamax
from keras.models import Model

if tf.executing_eagerly():
    tf.compat.v1.disable_eager_execution()

IMG_SIZE = 224

EPOCHS = 50

BATCH_SIZE = 32

DEBUG = False

DIVIDER = 1

SAVE_MODEL = True

TRAIN_END = 20000
TEST_START = TRAIN_END + 1
if DEBUG:
    EPOCHS = 1
    DIVIDER = BATCH_SIZE
    TRAIN_END = 600
    TEST_START = TRAIN_END + 1


def get_vgg16_output(vgg16, array_input, n_maps):
    picture_train_features = vgg16.predict(array_input)

    feature_map = np.empty([n_maps, 512])
    for idx_pic, picture in enumerate(picture_train_features):
        feature_map[idx_pic] = picture
    print("\n\n", feature_map.shape)
    return feature_map


def split_for_test(list):
    train = list[0:TRAIN_END]
    test = list[TEST_START:]
    return train, test


def main():

    trdata = ImageDataGenerator(rescale=1./255)

    traindata = trdata.flow_from_directory(
        directory="data/train", target_size=(IMG_SIZE, IMG_SIZE))

    tsdata = ImageDataGenerator(rescale=1./255)
    testdata = tsdata.flow_from_directory(
        directory="data/test", target_size=(IMG_SIZE, IMG_SIZE))


    traindata.reset()
    x_data, y_data = next(traindata)

    for i in range(int(len(traindata)/DIVIDER)-1):
        img, label = next(traindata)
        x_data = np.append(x_data, img, axis=0)
        y_data = np.append(y_data, label, axis=0)
    print("\n TRAIN DATA SHAPE: ", x_data.shape, y_data.shape, "\n")

    x_train, x_test = split_for_test(x_data)
    y_train, y_test = split_for_test(y_data)

    n_train = int(len(x_train))
    n_test = int(len(x_test))

    vgg16 = VGG16(
        include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3), pooling="avg", weights="imagenet"
    )

    x_train_feature_map = get_vgg16_output(vgg16, x_train, n_train)
    x_test_feature_map = get_vgg16_output(vgg16, x_test, n_test)

    top_layer_model = Sequential()
    top_layer_model.add(Dense(256, input_shape=(512,), activation="relu"))
    top_layer_model.add(Dense(256, input_shape=(256,), activation="relu"))
    top_layer_model.add(Dropout(0.5))
    top_layer_model.add(Dense(128, input_shape=(256,)))
    top_layer_model.add(Dense(2, activation="softmax"))

    adamax = Adamax()

    top_layer_model.compile(
        loss="categorical_crossentropy", optimizer=adamax, metrics=["accuracy"]
    )

    top_layer_model.fit(
        x_train_feature_map,
        y_train,
        validation_data=(x_test_feature_map, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE
    )

    score = top_layer_model.evaluate(x_test_feature_map, y_test, batch_size=BATCH_SIZE)

    print("Top layer model training on test: {}".format(score))

    inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    vg_output = vgg16(inputs)

    print("vg_output: {}".format(vg_output.shape))

    model_predictions = top_layer_model(vg_output)

    final_model = Model(inputs=inputs, outputs=model_predictions)

    final_model.compile(
        loss="categorical_crossentropy", optimizer=adamax, metrics=["accuracy"]
    )

    final_model_score = final_model.evaluate(
        x_train, y_train, batch_size=BATCH_SIZE
    )
    print("final_model (train score): {}".format(final_model_score))

    final_model_score = final_model.evaluate(
        x_test, y_test, batch_size=BATCH_SIZE
    )

    print("final_model (test score): {}".format(final_model_score))

    if SAVE_MODEL:
        final_model.save("/export")
    """
    data layers
    5: rgb value
    4: 224 image
    3: images
    2: batch sizes
    1: 
    """


if __name__ == "__main__":
    main()
