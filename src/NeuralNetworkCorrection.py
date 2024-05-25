import random

import numpy as np
import tensorflow as tf
from keras.src.optimizers.adam import Adam

from src import Utils, HammingCode

CHUNK_SIZE = 7

def get_training_data_set(sample_length, variancia):

    data = np.random.choice([0, 1], size=int(sample_length))
    split_encoded_data, split_original_data, merged_encoded_data = HammingCode.encode_sample(data)

    # apply noise
    ruido = np.random.normal(0, np.sqrt(variancia/2), size=int(len(merged_encoded_data)))
    noisy_encoded_data = np.array(merged_encoded_data) + ruido

    return {
        'ruido': ruido,
        'split_encoded_data': noisy_encoded_data, # will only split after tempering with eb_db
        'split_original_data': split_original_data,
        'encoded_data': noisy_encoded_data,
        'original_data': data
    }

def generate_data_for_training(training_data_set, Eb_db):

    Eb = 10**(Eb_db /10)

    amostra_ruidosa = np.sqrt(Eb) * np.array(training_data_set['encoded_data'])

    amostra_ruidosa_digital = [1 if x > 0.0 else 0 for x in amostra_ruidosa]

    normalizedInfo = training_data_set

    normalizedInfo['encoded_data'] = amostra_ruidosa_digital
    normalizedInfo['split_encoded_data'] = np.array_split(amostra_ruidosa_digital, int(len(amostra_ruidosa_digital) / 7))

    return normalizedInfo

def train_neural_network(training_data, epoches, batch_size):


    model = tf.keras.Sequential([
        tf.keras.layers.Dense(7, activation='relu', input_shape=(7,)),
        tf.keras.layers.Dense(7, activation='relu'),
        tf.keras.layers.Dense(4, activation='sigmoid')
    ])

    print(all(len(each) == 7 for each in training_data['noisy']))
    print(all(len(each) == 4 for each in training_data['original']))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(
        training_data['noisy'],
        training_data['original'],
        epochs=epoches,
        batch_size=batch_size,
        validation_data=(training_data['noisy'], training_data['original'])
    )

    return model

def decode_and_correct(encoded_data, model):
    decoded_data = []
    for input_index in range(0, len(encoded_data), 7):
        encoded_array = np.array(Utils.toInt(encoded_data[input_index:input_index + 7]))
        encoded_array = tf.expand_dims(encoded_array, axis=0)
        decoded_data.extend(Utils.roundToBits((model.predict(encoded_array)[0])))
    return decoded_data
