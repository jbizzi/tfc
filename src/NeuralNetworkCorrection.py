import copy
import random

import numpy as np
import tensorflow as tf
from keras import Sequential
from keras.src.layers import SimpleRNN, Dense, Dropout
import math

from sklearn.model_selection import train_test_split

from src import Utils, HammingCode

CHUNK_SIZE = 7

def get_training_data_set(sample_length):

    data = np.random.choice([0, 1], size=int(sample_length))

    split_encoded_data, split_original_data, merged_encoded_data = HammingCode.encode_sample(data)

    merged_encoded_data = [1.0 if bit == 1 else -1.0 for bit in merged_encoded_data]
    return {
        'split_encoded_data': [], # will only split after tempering with eb_db
        'split_original_data': split_original_data,
        'encoded_data': merged_encoded_data,
        'original_data': data,
        'noisy_original_data_15_11': [],
        'split_noisy_original_data_15_11': [],
        'split_original_data_15_11': []
    }

def generate_data_for_training(training_data_set, Eb_db, variancia):

    Eb_7_4 = 10**((Eb_db + 10*np.log10(4/7)) / 10)
    Eb = 10**((Eb_db) / 10)

    # apply noise for encoded data
    ruido_encoded = np.random.normal(0, np.sqrt(variancia/2), size=int(len(training_data_set['encoded_data'])))
    amostra_ruidosa_7_4 = np.sqrt(Eb_7_4) * np.array(training_data_set['encoded_data']) + ruido_encoded
    amostra_ruidosa_digital_7_4 = [1 if x > 0.0 else 0 for x in amostra_ruidosa_7_4]

    ruido_original = np.random.normal(0, np.sqrt(variancia/2), size=int(len(training_data_set['original_data'])))
    amostra_ruidosa_original = np.sqrt(Eb) * np.array(training_data_set['original_data']) + ruido_original
    amostra_ruidosa_original_digital = [1 if x > 0.0 else 0 for x in amostra_ruidosa_original]

    normalizedInfo = copy.deepcopy(training_data_set)

    normalizedInfo['encoded_data'] = amostra_ruidosa_digital_7_4
    normalizedInfo['split_encoded_data'] = split_and_pad(7, amostra_ruidosa_digital_7_4)

    normalizedInfo['split_original_data_15_11'] = split_and_pad(11, training_data_set['original_data'])
    normalizedInfo['split_noisy_original_data_15_11'] = split_and_pad(11, amostra_ruidosa_original_digital)

    return normalizedInfo

def split_and_pad(chunk, array):


    total_chunks = math.floor(len(array) / 11)

    total_length_chunks = total_chunks * 11
    smaller_array_len = len(array) - total_length_chunks
    remaining = 11 - smaller_array_len

    padded_array = np.pad(array, (0, remaining), 'constant', constant_values=0)
    return np.array_split(padded_array, int(len(padded_array) / 11))

def train_neural_network(training_data, epoches, batch_size):

    model = tf.keras.Sequential()

    model.add(Dense(512, input_dim=11, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(11, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    X_train, X_test, y_train, y_test = train_test_split(
        training_data['noisy'],
        training_data['original'],
        test_size=0.1,
        random_state=42
    )

    model.fit(
        X_train,
        y_train,
        epochs=epoches,
        batch_size=batch_size,
        validation_data=(X_test, y_test)
    )

    return model

def decode_and_correct(encoded, model):
    decoded_data = []
    for index in range(0, len(encoded), 100):
        decoded = Utils.roundToBits(model.predict_on_batch(np.array(encoded[index:index + 100])))
        decoded_data.extend(decoded)

    return np.array(decoded_data).flatten()
