# Dogs and Cats ML
Machine Learning model built to identify pictures between cats and dogs. Built with tensorflow and keras.

## Layers
The ML Model is built with a base model of VGG16 to determine features of the images and then a top layer model is used to dense down the features in a decision tree to classify the image. The top layer model features 2 Dense layers with a Dropout layer to leave the outliers followed by 2 more dense layers. The final dense layer uses an activation function of softmax to bring the data back to standard and classify the image as cat or dog.

## Accuracy
The Model was able to achieve a final test accuracy of 93.3 when evaluating the test data.

## Layout

data is split into train and test and then dogs and cats. export folder contains the saved model. `load.py` contains the loaded model from the export folder you can run.

```
/data
    /train
        /dogs
        /cats
    /test
        /test
/export
```

## Data
Data is from the dogs-vs-cats dataset on [kaggle](https://www.kaggle.com/c/dogs-vs-cats/data) or `kaggle competitions download -c dogs-vs-cats` with the kaggle command line

## Installation
Install Python 3.8+

```bash
pip install tensorflow keras numpy
```

## License
[MIT](https://choosealicense.com/licenses/mit/)
