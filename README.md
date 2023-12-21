# WebApp predicting Sentiment140 tweets
This app can predict sentiment from kaggle Dataset Sentiment140 tweets or for any tweets.

# Context
This was a `kaggle competition` held several years ago.
See [Sentiment140_dataset](https://www.kaggle.com/datasets/kazanova/sentiment140) for much more detailed documentation.

# About the app
It is in fact a python `Unidirectional LSTM Word2Vec trained on raw text` model.h5 with a tokenizer.pkl.

# Librairies
You need to install first some python librairies:

```
import re, os, time
from flask import Flask, request, jsonify, send_file
import tensorflow as tf
import numpy as np
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('snowball_data')
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import pickle
```

# Goal
As you know now, our main objective is to predict tweets sentiment although our model was only trained with the `Sentiment140 dataset`.

## Target
* 0 for a negative sentiment
* 1 for a positive sentiment

Our function `predict_sentiment` gives us a probability score.

```
if probability_score < 0.5:
        sentiment = "negative"
    else:
        sentiment = "positive"
```
## Recommandations
I strongly recommand you to install packages from the `requirements.txt`.

# Tests section
This section was created to test our app (unit tests) with `pytest`.