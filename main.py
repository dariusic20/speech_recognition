from functii import generate_graph, build_model, get_data_splits, load_dataset, preprocess
import os
import datetime
from tensorflow.keras.callbacks import Callback, TensorBoard, ModelCheckpoint
import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.utils import plot_model
import librosa
import tempfile
import pickle
import time
import keras
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from matplotlib.animation import FuncAnimation
import sounddevice as sd
import wavio
import pandas as pd
import base64


class PlotLosses(Callback):
    def __init__(self):
        self.train_loss_values = []
        self.val_loss_values = []
        self.train_accuracy_values = []
        self.val_accuracy_values = []
        self.train_chart = st.line_chart()
        self.val_chart = st.line_chart()
        self.last_update_time = 0

    def on_epoch_end(self, epoch, logs={}):
        self.train_loss_values.append(logs.get('loss'))
        self.val_loss_values.append(logs.get('val_loss'))
        self.train_accuracy_values.append(logs.get('accuracy'))
        self.val_accuracy_values.append(logs.get('val_accuracy'))

        current_time = time.time()
        val_loss_color = '#FF0000'
        val_accuracy_color = '#FFA500'
        train_accuracy_color = '#ADD8E6'
        train_loss_color = '#0000FF'
        if current_time - self.last_update_time > 0.1:
            self.train_chart.line_chart({'Training Loss': self.train_loss_values,
                                         'Validation Loss': self.val_loss_values}, color=[train_loss_color,
                                                                                          val_loss_color])

            self.val_chart.line_chart({'Training Accuracy': self.train_accuracy_values,
                                       'Validation Accuracy': self.val_accuracy_values}, color=[train_accuracy_color,
                                                                                                val_accuracy_color])
            self.last_update_time = current_time


def save_parameters(model_name, conv_params, dense_params):
    data = {
        "model_name": model_name,
        "conv_params": conv_params,
        "dense_params": dense_params,
    }
    with open("model_parameters.pkl", "wb") as file:
        pickle.dump(data, file)

def load_parameters():
    try:
        with open("model_parameters.pkl", "rb") as file:
            data = pickle.load(file)
            return data
    except FileNotFoundError:
        return None

def page1():
    st.write("Pagina 1")
    if st.button('Incarcarea datelor'):
        image_path = generate_graph()
        load_dataset("data/data.json")
        st.image(image_path, use_column_width=True)

def page2():
    st.write("Construirea modelului")

    model_name = st.text_input("Numele modelului")

    num_conv_layers = st.number_input("Numărul de straturi de convoluție", min_value=1, max_value=10, value=1,
                                      key="num_conv_layers")

    conv_params = []

    for i in range(num_conv_layers):
        st.subheader(f"Parametri pentru stratul de convoluție {i + 1}")
        conv_neurons = st.number_input(f"Numărul de filtre pentru stratul {i + 1}", min_value=1, max_value=1000,
                                       value=64, key=f"conv_neurons_{i}")
        conv_filter_size = st.slider("Dimensiunea filtrelor", min_value=1, max_value=3, value=3,
                                     key=f"conv_filter_size_{i}")
        conv_activation = st.selectbox("Funcție de activare", ["relu", "sigmoid", "tanh"], key=f"conv_activation_{i}")
        conv_padding = st.radio("Padding", ["same", "valid"], key=f"conv_padding_{i}")
        conv_stride = st.slider("Stride", min_value=1, max_value=3, value=2, key=f"conv_stride_{i}")
        conv_params.append((conv_filter_size, conv_activation, conv_padding, conv_stride, conv_neurons))

    num_dense_layers = st.number_input("Numărul de straturi Dense", min_value=1, max_value=10, value=1,
                                       key="num_dense_layers")

    dense_params = []

    for i in range(num_dense_layers):
        st.subheader(f"Parametri pentru stratul Dense {i + 1}")
        dense_neurons = st.number_input(f"Numărul de neuroni pentru stratul {i + 1}", min_value=1, max_value=1000,
                                        value=64, key=f"dense_neurons_{i}")
        dense_activation = st.selectbox("Funcție de activare", ["relu", "sigmoid", "tanh"], key=f"dense_activation_{i}")
        dropout_rate = st.slider("Rata de Dropout", min_value=0.0, max_value=0.9, value=0.3, step=0.01,
                                 key=f"dropout_rate_{i}")
        dense_params.append((dense_neurons, dense_activation, dropout_rate))

    if st.button("Salveaza parametrii"):
        save_parameters(model_name, conv_params, dense_params)
        st.write("Parametrii salvati")

        st.subheader("Parametrii salvati:")
        st.write(f"Numele modelului: {model_name}")
        st.write("Parametrii pentru straturile de convoluție:")
        for i, (conv_filter_size, conv_activation, conv_padding, conv_stride, conv_neurons) in enumerate(conv_params,
                                                                                                         start=1):
            st.write(
                f"Stratul {i}: Filtre: {conv_filter_size}, Activare: {conv_activation}, Padding: {conv_padding}, Stride: {conv_stride}, Neuroni: {conv_neurons}")
        st.write("Parametrii pentru straturile Dense:")
        for i, (dense_neurons, dense_activation, dropout_rate) in enumerate(dense_params, start=1):
            st.write(f"Stratul {i}: Neuroni: {dense_neurons}, Activare: {dense_activation}, Dropout: {dropout_rate}")


def page3():
    st.write("Pagina 3")
    model_params = load_parameters()

    st.write(f"Numele modelului: {model_params['model_name']}")

    for i, conv_param in enumerate(model_params['conv_params'], start=1):
        st.write(f"Stratul de convoluție {i}: {conv_param}")
    st.write("Parametrii pentru fiecare strat Dense:")
    for i, dense_param in enumerate(model_params['dense_params'], start=1):
        st.write(f"Stratul Dense {i}: {dense_param}")

    st.subheader("Parametrii pentru antrenare")
    learning_rate = st.number_input("Rata de învățare", min_value=0.00001, max_value=1.0, value=0.00010, step=0.00001, format="%.5f")
    num_epochs = st.number_input("Numărul de epoci", min_value=1, max_value=100, value=10)
    batch_size = st.number_input("Dimensiunea lotului", min_value=1, max_value=128, value=32)

    if st.button("Antrenează modelul"):
        st.write("Incepe procesul de antrenare")
        X_train, X_validation, X_test, y_train, y_validation, y_test = get_data_splits("data/data.json")

        input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])

        model = build_model(model_params['model_name'], model_params['conv_params'], model_params['dense_params'], input_shape, learning_rate)

        log_dir = os.path.join("logs", "fit", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
        checkpoint_callback = ModelCheckpoint(filepath="model_checkpoint.h5", save_best_only=True)

        callbacks_list = [tensorboard_callback, checkpoint_callback, PlotLosses()]
        history = model.fit(X_train, y_train, epochs=num_epochs, batch_size=batch_size,
                  validation_data=(X_validation, y_validation), callbacks=callbacks_list)

        model.save("model.h5")

        accuracy = history.history['accuracy']
        val_accuracy = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']

        file_path = os.path.join(log_dir, "training_metrics.txt")
        metrics = [
            ("accuracy", accuracy),
            ("val_accuracy", val_accuracy),
            ("loss", loss),
            ("val_loss", val_loss),
        ]

        with open(file_path, 'w') as file:
            for metric_name, metric_values in metrics:
                if not isinstance(metric_values, list):
                    metric_values = [metric_values]

                metric_values_str = ', '.join(map(str, metric_values))
                file.write(f"{metric_name}: {metric_values_str}\n")



def register_audio(audio_length=1, sampling_rate=44100):
    st.write("Apasă butonul pentru a începe înregistrarea...")
    if st.button("Înregistrează"):
        st.write("Înregistrare începută...")
        record = sd.rec(int(audio_length * sampling_rate), samplerate=sampling_rate, channels=1, dtype='int16')
        sd.wait()

        file_name = "înregistrare_audio.wav"
        wavio.write(file_name, record, sampling_rate, sampwidth=2)

        return file_name

_mappings = [
    "backward",
    "bed",
    "bird",
    "cat",
    "dog",
    "down",
    "eight",
    "five",
    "follow",
    "forward",
    "four",
    "go",
    "happy",
    "house",
    "learn",
    "left",
    "marvin",
    "nine",
    "no",
    "off",
    "on",
    "one",
    "right",
    "seven",
    "sheila",
    "six",
    "stop",
    "three",
    "tree",
    "two",
    "up",
    "visual",
    "wow",
    "yes",
    "zero"
]


def show_spectrogram(audio_path):
    y, sr = librosa.load(audio_path)
    D = np.abs(librosa.stft(y))

    fig, ax = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max), sr=sr, x_axis='time', y_axis='log', ax=ax)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    plt.title('Spectrograma')
    plt.xlabel('Timp')
    plt.ylabel('Frecvență')
    st.pyplot(fig)

def show_prediction_results(probabilities):
    predicted_index = np.argmax(probabilities)
    predicted_keyword = _mappings[predicted_index]

    max_probability = np.max(probabilities)
    confidence_threshold = 0.45

    if max_probability < confidence_threshold:
        prediction_result = "Nesigur"
    else:
        prediction_result = predicted_keyword

    if max_probability >= confidence_threshold:
        st.header("🔔Predictia:")
        st.header(prediction_result)

    else:
        st.write("❓ Modelul este nesigur cu privire la predicție.")


def page4():
    cuvinte = [
        "backward", "bed", "bird", "cat", "dog", "down", "eight", "five", "follow", "forward",
        "four", "go", "happy", "house", "learn", "left", "marvin", "nine", "no", "off", "on",
        "one", "right", "seven", "sheila", "six", "stop", "three", "tree", "two", "up",
        "visual", "wow", "yes", "zero"
    ]
    st.write("# Cuvinte pe care modelul le cunoaste:")
    st.table(cuvinte)
    audio_path = register_audio()
    st.audio(audio_path)
    show_spectrogram(audio_path)

    model = keras.models.load_model('model.h5')

    MFCCs = preprocess(audio_path)

    MFCCs = MFCCs[np.newaxis, ..., np.newaxis]

    predictions = model.predict(MFCCs)
    probabilities = np.squeeze((np.exp(predictions) / np.sum(np.exp(predictions), axis=-1)) * 10)
    show_prediction_results(probabilities)

selected_page = st.sidebar.selectbox(
    "Selectează pagina:",
    ("Incarcarea datelor", "Construirea modelului", "Antrenarea modelului", "Generarea predictiilor")
)

pages = {
    "Incarcarea datelor": page1,
    "Construirea modelului": page2,
    "Antrenarea modelului": page3,
    "Generarea predictiilor": page4
}

pages[selected_page]()



