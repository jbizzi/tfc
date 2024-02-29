import numpy as np
import tensorflow as tf

import numpy as np
import tensorflow as tf
from src import Utils

sample_length = 2 ** 4
chunk_size = 7
noise_rates = np.linspace(0, 1, 11)


def generate_data(num_samples):
    data = []
    for i in range(0, len(noise_rates)):
        data.append(''.join(np.random.choice(['0', '1'], size=sample_length)))

    encoded_data = [Utils.encode_sample(bit) for bit in data]
    noisy_data = [Utils.noiseString(error_rate, word) for word, error_rate in zip(encoded_data, noise_rates)]
    return noisy_data, data


# Função para criar e treinar a rede neural
def train_neural_network(noisy_data, original_data):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(7, activation='relu', input_shape=(7,)),  # Ajustado para o comprimento da string
        tf.keras.layers.Dense(7, activation='relu'),  # Ajustado para o comprimento da string
        tf.keras.layers.Dense(4, activation='sigmoid')  # Ajustado para o comprimento da string
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
        epochs=50,
        batch_size=int(sample_length / chunk_size)
    )

    return model

def decode_and_correct(encoded_data, model):
    decoded_data = []
    for input_index in range(0, len(encoded_data), 7):
        input = np.array(Utils.toInt(encoded_data[input_index:input_index + 7]))
        input = tf.expand_dims(input, axis=0)
        decoded_data.append(model.predict(input))
    decoded_data = np.round(decoded_data)
    return ''.join(str(int(each)) for each in np.concatenate(np.concatenate(decoded_data)))


# Gerar dados de treinamento
num_samples = 1000
string_length = 10  # Parâmetro para o comprimento das strings
noisy_data, original_data = generate_data(num_samples)

# Criar e treinar a rede neural
model = train_neural_network(noisy_data, original_data)

# Gerar dados de teste
test_noisy_data, test_original_data = generate_data(10)

# Decodificar e corrigir os bits usando a rede neural
decoded_data = decode_and_correct(test_noisy_data[0], model)

# Exibir os resultados
print("Bits originais:")
print(test_original_data[0])
print("Bits decodificados e corrigidos:")
print(decoded_data)


data = generate_data(1)
