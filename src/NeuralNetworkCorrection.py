import copy
import random

import numpy as np
import tensorflow as tf
from keras.src.optimizers.adam import Adam

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
        'original_data': data
    }

def generate_data_for_training(training_data_set, Eb_db, variancia):

    Eb = 10**(Eb_db /10)


    # apply noise
    ruido = np.random.normal(0, np.sqrt(variancia/2), size=int(len(training_data_set['encoded_data'])))

    amostra_ruidosa = np.sqrt(Eb) * np.array(training_data_set['encoded_data']) + ruido

    amostra_ruidosa_digital = [1 if x > 0.0 else 0 for x in amostra_ruidosa]

    normalizedInfo = copy.deepcopy(training_data_set)

    normalizedInfo['encoded_data'] = amostra_ruidosa_digital
    normalizedInfo['split_encoded_data'] = np.array_split(amostra_ruidosa_digital, int(len(amostra_ruidosa_digital) / 7))

    return normalizedInfo

def train_neural_network(training_data, epoches, batch_size):


    model = tf.keras.Sequential([
        tf.keras.layers.Dense(7, activation='relu', input_shape=(7,)),
        tf.keras.layers.Dense(7, activation='relu'),
        tf.keras.layers.Dense(4, activation='sigmoid')
    ])


    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(
        training_data['noisy'],
        training_data['original'],
        epochs=epoches,
        batch_size=batch_size,
        validation_data=(training_data['noisy'], training_data['original'])
    )

    return model

def decode_and_correct(encoded, model):
    decoded_data = []
    for chunk in encoded:
        encoded_array = tf.expand_dims(chunk, axis=0)
        decoded_data.extend(Utils.roundToBits((model.predict(encoded_array)[0])))
    return decoded_data
