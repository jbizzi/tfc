import random

import numpy as np
import tensorflow as tf
import tensorflow.keras as tfk
from keras.src.optimizers.adam import Adam

from src import Utils, HammingCode

CHUNK_SIZE = 7

def decode(model, full_sample):

    decoded = []
    for input_index in range(0, len(full_sample), 7):
        input = Utils.toInt(full_sample[input_index:input_index + 7])
        input = tf.expand_dims(input, axis=0)
        decoded_bits = model.predict(input)[0]
        decoded.extend(Utils.roundToBits(decoded_bits))
    return decoded

def create_and_train_auto_encoder(training_data, epoches, batch_size):

    autoencoder = create_auto_encoder()

    autoencoder.fit(
        training_data['noisy'],
        training_data['original'],
        epochs=epoches,
        batch_size=batch_size,
        validation_data=(training_data['noisy'], training_data['original'])
    )
    return autoencoder

def create_auto_encoder():

    input = tfk.layers.Input(shape=(7,))

    # encoder
    encoder = tfk.layers.Dense(7, activation='relu')(input)

    # decoder with encoder as input
    decoder = tfk.layers.Dense(4, activation='sigmoid')(encoder)

    # model
    autoencoder = tfk.models.Model(inputs=input, outputs=decoder)

    autoencoder.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return autoencoder

