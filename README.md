# Dogs and Cats ML
Machine Learning model built to identify pictures between cats and dogs. Built with tensorflow and keras.


## Layout

data is split into train and test and then dogs and cats. export folder contains the saved model. `load.py` contains the loaded model from the export folder you can run.

/data
    /train
        /dogs
        /cats
    /test
        /test
/export

## Data
Data is from the dogs-vs-cats dataset on [kaggle](https://www.kaggle.com/c/dogs-vs-cats/data) or `kaggle competitions download -c dogs-vs-cats` with the kaggle command line

## Installation
Install Python 3.8+

```bash
pip install tensorflow keras numpy
```

## License
[MIT](https://choosealicense.com/licenses/mit/)