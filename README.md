
# Dogs and Cats ML

Machine Learning model built to identify pictures between cats and dogs. Built
with tensorflow and keras.

## Images into data

using the Keras ImageDataGenerator the collection of the data was simple enough
with just a few lines

```py
trdata = ImageDataGenerator(rescale=1./255)
traindata = trdata.flow_from_directory(
    directory="data/train", target_size=(IMG_SIZE, IMG_SIZE))

tsdata = ImageDataGenerator(rescale=1./255)
testdata = tsdata.flow_from_directory(
    directory="data/test", target_size=(IMG_SIZE, IMG_SIZE))
```

## VGG16

This Machine Learning model uses the Keras VGG16 model weighted with imagenet to
extract features of the images. VGG16 uses Convulutional 2D Layers to extract
features.

## Model

The top layer model on top of the vgg16 model for this project was a Keras
Sequential Model.

```py
top_layer_model = Sequential()
```

### Layers

The top layer model consisted of 5 layers, 4 Dense and one Dropout layer. The
Dense layers form a decision tree to best decide how to classify the data while
the Dropout layer was to kill the outlier data to get a more normalized dataset.
The final dense layer uses an activation function of softmax to bring the data
back to standard and classify the image as cat or dog.

```py
top_layer_model.add(Dense(256, input_shape=(512,), activation="relu"))
top_layer_model.add(Dense(256, input_shape=(256,), activation="relu"))
top_layer_model.add(Dropout(0.5))
top_layer_model.add(Dense(128, input_shape=(256,)))
top_layer_model.add(Dense(2, activation="softmax"))
```

### Optimizer

This model uses an Adamax optimizer from Keras

### Loss

This model uses categorical_crossentropy loss function to penalize the model

```py
top_layer_model.compile(
        loss="categorical_crossentropy", optimizer=adamax, metrics=["accuracy"]
    )
```

## Combining the Models

To Combine the VGG16 and top layer model I used Keras' Model

```py
inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
vg_output = vgg16(inputs)

model_predictions = top_layer_model(vg_output)
final_model = Model(inputs=inputs, outputs=model_predictions)

final_model.compile(
    loss="categorical_crossentropy", optimizer=adamax, metrics=["accuracy"]
    )
```

## Accuracy

The Model was able to achieve a final test accuracy of 93.3% when evaluating the
test data.

```py
loss, acc = final_model.evaluate(
        x_test, y_test, batch_size=BATCH_SIZE
    )

print("final_model (test score) accuracy: {}".format(acc))
```

## Layout

data is split into train and test and then dogs and cats. export folder contains
the saved model. load.py contains the loaded model from the export folder you
can run.

```py
/data
    /train
        /dogs
        /cats
    /test
        /test
/export
```

## Data

Data is from the dogs-vs-cats dataset on kaggle or

`kaggle competitions download -c dogs-vs-cats`

with the kaggle command line


## Installation
Install Python 3.8+

```bash
pip install tensorflow keras numpy
```

## License
[MIT](https://choosealicense.com/licenses/mit/)
