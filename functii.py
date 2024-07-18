import matplotlib.pyplot as plt
import os
import librosa
import seaborn as sns
import tempfile
import numpy as np
import tensorflow as tf
import keras
import json
import datetime
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint


def generate_graph():
    dataset_path = "data/speech_commands_v0.02"
    categories = []
    files_count = []

    name_to_number = {}
    count = 0

    for root, dirs, files in os.walk(dataset_path):
        if len(files) > 0:
            category = os.path.basename(root)
            categories.append(category)
            files_count.append(len(files))
            name_to_number[category] = count
            count += 1

    sns.set_style("darkgrid")
    plt.style.use('bmh')
    fig, ax = plt.subplots(figsize=(12, 8))

    bars = ax.bar(range(len(categories)), files_count)
    ax.set_xlabel('Clasa')
    ax.set_ylabel('Număr de fișiere')
    ax.set_title('Numărul total de fișiere pentru fiecare clasă')
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(range(len(categories)), rotation=0)

    legend_labels = [f'{i} - {cat}' for i, cat in enumerate(categories)]
    ax.legend(bars, legend_labels, loc='center left', bbox_to_anchor=(1, 0.5), title="Clase")

    plt.tight_layout()

    for bar, num_files, category in zip(bars, files_count, categories):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height / 2, f'{num_files}', ha='center', va='center', rotation=90,
                color='black')

    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_file:
        plt.savefig(temp_file.name, format='png')

    plt.close()

    return temp_file.name


def load_dataset(data_path):
    with open(data_path, "r") as fp:
        data = json.load(fp)

    X = np.array(data["MFCCs"])
    y = np.array(data["labels"])

    return X, y

def get_data_splits(data_path, test_size = 0.1, test_validation = 0.1 ):

    X, y = load_dataset(data_path)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=test_validation)

    X_train = X_train[..., np.newaxis]
    X_validation = X_validation[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    return X_train, X_validation, X_test, y_train, y_validation, y_test

def preprocess(file_path, n_mfcc=13, n_fft = 2048, hop_length=512):
    signal, sr = librosa.load(file_path)
    if len(signal) > 22050:
        signal = signal[:22050]
    MFCCs = librosa.feature.mfcc(y=signal, n_mfcc=n_mfcc, n_fft = n_fft, hop_length=hop_length)
    return MFCCs.T

def build_model(name, conv_params, dense_params, input_shape, learning_rate):
    model = keras.Sequential(name=name)

    for i, (conv_filter_size, conv_activation, conv_padding, conv_stride, conv_neurons) in enumerate(conv_params,
                                                                                                     start=1):
        if i == 1:
            model.add(
                keras.layers.Conv2D(conv_neurons, (conv_filter_size, conv_filter_size), activation=conv_activation,
                                    input_shape=input_shape,
                                    kernel_regularizer=keras.regularizers.l2(0.001)))
        else:
            model.add(
                keras.layers.Conv2D(conv_neurons, (conv_filter_size, conv_filter_size), activation=conv_activation,
                                    kernel_regularizer=keras.regularizers.l2(0.001)))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.MaxPooling2D((conv_filter_size, conv_filter_size), strides=(conv_stride, conv_stride),
                                            padding=conv_padding))

    model.add(keras.layers.Flatten())
    for i, (neurons, activation, dropout_rate) in enumerate(dense_params, start=1):
        model.add(keras.layers.Dense(neurons, activation=activation))
        model.add(keras.layers.Dropout(dropout_rate))

    model.add(keras.layers.Dense(35, activation="softmax"))

    optimizer = keras.optimizers.Adam(learning_rate = learning_rate)
    model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    return model
