import random

import numpy as np
import tensorflow as tf
from src import Utils

CHUNK_SIZE = 7


def getDefaultData():
    return np.array((
        [False, False, False, False],
        [False, False, False, True],
        [False, False, True, False],
        [False, False, True, True],
        [False, True, False, False],
        [False, True, False, True],
        [False, True, True, False],
        [False, True, True, True],
        [True, False, False, False],
        [True, False, False, True],
        [True, False, True, False],
        [True, False, True, True],
        [True, True, False, False],
        [True, True, False, True],
        [True, True, True, False],
        [True, True, True, True],
    ))


def generate_data_for_training(noise_rates, total_samples):


    data = getDefaultData()
    if (total_samples < len(data)): # not enough samples

    encoded_data = [Utils.encode_sample(bit) for bit in data]



    noisy_data = [Utils.noiseString(error_rate, word) for word, error_rate in zip(encoded_data, noise_rates)]
    return noisy_data, data


# Função para criar e treinar a rede neural
def train_neural_network(noisy_data, original_data, epoches, sample_length):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(7, activation='relu', input_shape=(7,)),
        tf.keras.layers.Dense(7, activation='relu'),
        tf.keras.layers.Dense(4, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # reshape data to be 7 bits and 4 bits each
    noisy_data_reshaped = []
    for noisy_sample in noisy_data:
        for index in range(0, len(noisy_sample), 7):
            noisy_data_reshaped.append(Utils.toInt(noisy_sample[index:index + 7]))
    noisy_data_reshaped = np.array(noisy_data_reshaped)

    original_data_reshaped = []
    for original_sample in original_data:
        for index in range(0, len(original_sample), 4):
            original_data_reshaped.append(np.array(Utils.toInt(list(original_sample)[index:index + 4]), dtype=int))
    original_data_reshaped = np.array(original_data_reshaped)

    model.fit(
        noisy_data_reshaped,
        original_data_reshaped,
        epochs=epoches,
        batch_size=int(sample_length / CHUNK_SIZE)
    )
    return model


def generate_data_for_testing(word_length, noise_rates, words):
    data = []
    for i in range(0, words):
        data.append(np.random.choice([True, False], size=word_length))

    encoded_data = [Utils.encode_sample(bit) for bit in data]

    noisy_data = [Utils.noiseString(random.randint(0, 100), word) for word in encoded_data]
    return noisy_data, data


def decode_and_correct(encoded_data, model):
    decoded_data = []
    for input_index in range(0, len(encoded_data), 7):
        input = np.array(Utils.toInt(encoded_data[input_index:input_index + 7]))
        input = tf.expand_dims(input, axis=0)
        decoded_data.append(model.predict(input))
    decoded_data = np.round(decoded_data)
    return ''.join(str(int(each)) for each in np.concatenate(np.concatenate(decoded_data)))


generate_data_for_training(np.linspace(0, 1, 101))
