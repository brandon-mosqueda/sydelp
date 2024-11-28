import keras
import re
import pandas as pd
import requests as rq

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.preprocessing.text import Tokenizer  # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences  # type: ignore

from os.path import isfile
from zipfile import ZipFile
from io import BytesIO

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

from typing import Union
from utils.typing import NumArray, IntArray

load_mnist = keras.datasets.mnist.load_data

class TextCleaner:
    lemmatizer: WordNetLemmatizer
    stop_words: set

    def __init__(self) -> None:
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

    def clean(self, text) -> str:
        # Removing non-word characters
        text = re.sub(r'\W', ' ', text).lower()

        cleaned_text: list[str] = [
            self.lemmatizer.lemmatize(word)
            for word in text.split()
            if word not in self.stop_words
        ]

        return ' '.join(cleaned_text)


def iris_data(testing_proportion: float = 0.2) -> tuple[NumArray, NumArray,
                                                        IntArray, IntArray]:
    # Load and preprocess the data
    X: NumArray
    y: IntArray
    X, y = load_iris(return_X_y=True)  # type: ignore

    X_train: NumArray
    X_test: NumArray
    y_train: IntArray
    y_test: IntArray

    # Split the data into balanced training and testing datasets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=testing_proportion, stratify=y
    )

    # Standardize the features
    scaler: StandardScaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test


def mnist_data() -> tuple[NumArray, NumArray, IntArray, IntArray]:
    # Load and preprocess the data
    X_train: NumArray
    X_test: NumArray
    y_train: IntArray
    y_test: IntArray
    (X_train, y_train), (X_test, y_test) = load_mnist()

    X_train = X_train.reshape(60000, 784).astype("float") / 255
    X_test = X_test.reshape(10000, 784).astype("float") / 255

    return X_train, X_test, y_train, y_test


def spam_data(testing_proportion: float = 0.2,
              vocabulary_size: int = 1000,
              max_sequence_length: int = 50,
              seed: Union[int, None] = None) -> tuple[NumArray, NumArray,
                                                      IntArray, IntArray]:
    file_path: str = "data/sms_spam_collection/SMSSpamCollection"

    if not isfile(file_path):
        url: str = (
            "https://archive.ics.uci.edu/ml/machine-learning-databases/"
            "00228/smsspamcollection.zip"
        )
        response = rq.get(url)
        if response.status_code == 200:
            with ZipFile(BytesIO(response.content)) as zip_ref:
                zip_ref.extractall("data/sms_spam_collection")
        else:
            raise ConnectionError("Failed to download file. Status code: %s" %
                                  response.status_code)

        Data: pd.DataFrame = pd.read_csv(file_path,
                                         sep='\t',
                                         header=None,
                                         names=['label', 'text'],
                                         encoding='latin-1')
        Data['label'] = LabelEncoder().fit_transform(Data['label'])
        cleaner: TextCleaner = TextCleaner()
        Data['text'] = Data['text'].apply(cleaner.clean)
        Data.to_csv("data/sms_spam_collection/SPAMcleaned.csv",
                    index=False,
                    header=True)
    else:
        Data: pd.DataFrame = pd.read_csv(
            "data/sms_spam_collection/SPAMcleaned.csv")
        Data['text'] = Data['text'].apply(str)

    X_train: NumArray
    X_test: NumArray
    y_train: IntArray
    y_test: IntArray

    X_train, X_test, y_train, y_test = train_test_split(
        Data['text'],
        Data['label'].to_numpy(),
        stratify=Data['label'],
        test_size=testing_proportion,
        random_state=seed
    )

    tokenizer = Tokenizer(num_words=vocabulary_size, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train)
    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)

    X_train_padded: NumArray = pad_sequences(
        X_train_seq,
        maxlen=max_sequence_length,
        padding='post',
        truncating='post'
    )
    X_test_padded: NumArray = pad_sequences(
        X_test_seq,
        maxlen=max_sequence_length,
        padding='post',
        truncating='post'
    )

    return X_train_padded, X_test_padded, y_train, y_test
