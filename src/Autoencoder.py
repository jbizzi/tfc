import random

import numpy as np
import tensorflow as tf
import tensorflow.keras as tfk
from keras import Input, Model
from keras.src.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.src.layers import Dense, BatchNormalization, GaussianNoise

from src import Utils, HammingCode

CHUNK_SIZE = 7
from src import NeuralNetworkCorrection

data_bits = NeuralNetworkCorrection.data_bits


def decode(model, full_sample):

    decoded = []
    for input_index in range(0, len(full_sample), 4):
        input = Utils.toInt(full_sample[input_index:input_index + 4])
        input = tf.expand_dims(input, axis=0)
        decoded_bits = model.predict(input)[0]
        decoded.extend(Utils.roundToBits(decoded_bits))
    return decoded

def create_and_train_auto_encoder(training_data):

    size = 4
    code_rate = 4/7
    neurons = 7
    entrada = Input(shape=(size,))
    encoder1 = Dense(size, activation='relu')(entrada)
    encoder2 = Dense(neurons, activation='linear')(encoder1)
    encoder3 = BatchNormalization()(encoder2)


    Eb_treinamento = 5.01187 * code_rate

    noise = GaussianNoise(np.sqrt(Eb_treinamento))(encoder3)
    decoder1 = Dense(size, activation='relu')(noise)
    decoder2 = Dense(size, activation='softmax')(decoder1)

    autoencoder = Model(entrada, decoder2)
    autoencoder.compile(optimizer='adam', loss='categorical_crossentropy')

    autoencoder.fit(
        training_data['original'],
        training_data['original'],
        epochs=20,
        batch_size=300,
        validation_data=(training_data['original'], training_data['original']),
    )

    # Com o modelo treinado, separa em encoder e decoder
    encoder = Model(entrada, encoder3)

    entrada_encoder = Input(shape=(7,))
    decoder_layer1 = autoencoder.layers[-2](entrada_encoder)
    decoder_layer2 = autoencoder.layers[-1](decoder_layer1)
    decoder = Model(entrada_encoder, decoder_layer2)
    return autoencoder, encoder, decoder
