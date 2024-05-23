import random

import numpy as np
import tensorflow as tf
from src import Utils, HammingCode

CHUNK_SIZE = 7

def getDefaultData():
    return np.array((
        [0, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 0, 1, 1],
        [0, 1, 0, 0],
        [0, 1, 0, 1],
        [0, 1, 1, 0],
        [0, 1, 1, 1],
        [1, 0, 0, 0],
        [1, 0, 0, 1],
        [1, 0, 1, 0],
        [1, 0, 1, 1],
        [1, 1, 0, 0],
        [1, 1, 0, 1],
        [1, 1, 1, 0],
        [1, 1, 1, 1],
    ))

def decode_sample(data):
    decoded_data = []
    for i in range(0, len(data) - 6, 7):
        decoded_data.extend(HammingCode.decode(data[i:i + 7]).T)
    return decoded_data

def encode_sample(data):
    encoded_data = []
    for i in range(0, len(data) - 3, 4):
        encoded_data.extend(HammingCode.encode(data[i:i + 4]))
    return encoded_data

def generate_data_for_training(variancia, sample_length, Eb_db):

    amostra = np.random.choice([0, 1], size=int(sample_length))
    amostra_codificada = encode_sample(amostra)

    for i in range(len(amostra_codificada)):
        if amostra_codificada[i] == 0.0:
            amostra_codificada[i] = -1.0
        else:
            amostra_codificada[i] = 1.0

    Eb = 10**(Eb_db /10)

    ruido = np.random.normal(0, np.sqrt(variancia/2), size=int(len(amostra_codificada)))

    amostra_ruidosa = np.sqrt(Eb) * np.array(amostra_codificada) + ruido

    amostra_recebida = np.sign(amostra_ruidosa)

    for i in range(len(amostra_recebida)):
        if amostra_recebida[i] == -1:
            amostra_recebida[i] = 0
        else:
            amostra_recebida[i] = 1

    return amostra_recebida, amostra


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
    print(noisy_data_reshaped)
    print(original_data_reshaped)
    model.fit(
        noisy_data_reshaped,
        original_data_reshaped,
        epochs=epoches,
        batch_size=int(sample_length / CHUNK_SIZE)
    )
    return model

def decode_and_correct(encoded_data, model):
    decoded_data = []
    for input_index in range(0, len(encoded_data), 7):
        input = np.array(Utils.toInt(encoded_data[input_index:input_index + 7]))
        input = tf.expand_dims(input, axis=0)
        decoded_data.append(model.predict(input))

    decoded = []
    for each in decoded_data:
        for bit_array in each:
            for bit in bit_array:
                corrected_bit = 1 if bit >= 0.5 else 0
                decoded.append(corrected_bit)


    return decoded #''.join((int(each)) for each in np.concatenate(np.concatenate(decoded_data)))


