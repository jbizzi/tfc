import copy

import numpy as np
import tensorflow as tf
from keras.src.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.src.layers import SimpleRNN, Dense, Dropout
import math

from sklearn.model_selection import train_test_split

from src import Utils, HammingCode

CHUNK_SIZE = 7
data_bits = 4
code_error = 11/15
def get_training_data_set(sample_length):

    data = np.random.choice([0, 1], size=int(sample_length))
    padded_data = pad_array(data_bits, data)

    split_encoded_data, split_original_data, merged_encoded_data = HammingCode.encode_sample(data)

    merged_encoded_data = [1.0 if bit == 1 else -1.0 for bit in merged_encoded_data]
    transmitted_padded_data = [1.0 if bit == 1 else -1.0 for bit in padded_data]

    return {
        'split_encoded_data': [], # will only split after tempering with eb_db
        'split_original_data': split_original_data,
        'encoded_data': merged_encoded_data,
        'original_data': data,
        'noisy_original_data_15_11': [],
        'split_noisy_original_data_15_11': [],
        'split_original_data_15_11': [],
        'padded_array_original_data': padded_data,
        'transmited_padded_array_original_data': transmitted_padded_data
    }

def generate_data_for_training(training_data_set, Eb_db, variancia):

    Eb = 10**((Eb_db) / 10)

    # apply noise for encoded data
    ruido_encoded = np.random.normal(0, np.sqrt(variancia/2), size=int(len(training_data_set['encoded_data'])))
    amostra_ruidosa_7_4 = np.sqrt(Eb*4/7) * np.array(training_data_set['encoded_data']) + ruido_encoded
    amostra_ruidosa_digital_7_4 = [1 if x > 0.0 else 0 for x in amostra_ruidosa_7_4]

    ruido_original = np.random.normal(0, np.sqrt(variancia/2), size=int(len(training_data_set['transmited_padded_array_original_data'])))
    amostra_ruidosa_original = np.sqrt(Eb*code_error) * np.array(training_data_set['transmited_padded_array_original_data']) + ruido_original
    amostra_ruidosa_original_digital = [1 if x > 0.0 else 0 for x in amostra_ruidosa_original]

    normalizedInfo = copy.deepcopy(training_data_set)

    normalizedInfo['encoded_data'] = amostra_ruidosa_digital_7_4
    normalizedInfo['split_encoded_data'] = np.array_split(amostra_ruidosa_digital_7_4, int(len(amostra_ruidosa_digital_7_4)/7))

    normalizedInfo['split_original_data_15_11'] = np.array_split(training_data_set['padded_array_original_data'], int(len(training_data_set['padded_array_original_data'])/data_bits))
    normalizedInfo['split_noisy_original_data_15_11'] = np.array_split(amostra_ruidosa_original_digital, int(len(amostra_ruidosa_original_digital)/data_bits))
    return normalizedInfo

def pad_array(chunk, array):
    total_chunks = math.floor(len(array) / chunk)

    total_length_chunks = total_chunks * chunk
    smaller_array_len = len(array) - total_length_chunks
    remaining = chunk - smaller_array_len
    padded_array = np.pad(array, (0, remaining), 'constant', constant_values=0)
    return padded_array

def split_and_pad(chunk, array):
    padded_array = pad_array(chunk, array)
    return np.array_split(padded_array, int(len(padded_array) / chunk))

def train_neural_network(training_data, epoches, batch_size):

    model = tf.keras.Sequential()

    model.add(Dense(512, input_dim=data_bits, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(data_bits, activation='sigmoid'))
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

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
        validation_data=(X_test, y_test),
        callbacks=[reduce_lr, early_stopping]
    )

    loss, accuracy = model.evaluate(X_test, y_test)
    print(f'Loss: {loss}')
    print(f'Accuracy: {accuracy}')

    return model

def decode_and_correct(encoded, model):
    decoded_data = []
    for index in range(0, len(encoded), 1000):
        decoded = Utils.roundToBits(model.predict_on_batch(np.array(encoded[index:index + 1000])))
        decoded_data.extend(decoded)

    return np.array(decoded_data).flatten()
