import random

import numpy as np
import tensorflow as tf
import tensorflow.keras as tfk
from keras.src.callbacks import EarlyStopping, ReduceLROnPlateau

from src import Utils, HammingCode

CHUNK_SIZE = 7

def decode(model, full_sample):

    decoded = []
    for input_index in range(0, len(full_sample), 4):
        input = Utils.toInt(full_sample[input_index:input_index + 4])
        input = tf.expand_dims(input, axis=0)
        decoded_bits = model.predict(input)[0]
        decoded.extend(Utils.roundToBits(decoded_bits))
    return decoded

def create_and_train_auto_encoder(training_data, epoches, batch_size):

    autoencoder = create_auto_encoder()
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    autoencoder.fit(
        training_data['noisy'],
        training_data['original'],
        epochs=epoches,
        batch_size=batch_size,
        validation_data=(training_data['noisy'], training_data['original']),
        callbacks=[reduce_lr, early_stopping]
    )
    return autoencoder

def create_auto_encoder():

    input = tfk.layers.Input(shape=(11,))

    # encoder
    encoder = tfk.layers.Dense(15, activation='relu')(input)

    # decoder with encoder as input
    decoder = tfk.layers.Dense(11, activation='sigmoid')(encoder)

    # model
    autoencoder = tfk.models.Model(inputs=input, outputs=decoder)

    autoencoder.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return autoencoder