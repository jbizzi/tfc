import numpy as np
import tensorflow as tf

# Parâmetros do código de Hamming
k = 4  # Número de bits de informação
n = 7  # Número total de bits (incluindo os bits de paridade)
parity_check_matrix = np.array([[1, 1, 0, 1, 1, 0, 0],
                                [1, 0, 1, 1, 0, 1, 0],
                                [0, 1, 1, 1, 0, 0, 1]])

# Funções auxiliares
def hamming_encode(bits):
    bits_with_parity = np.zeros(n, dtype=int)
    bits_with_parity[:k] = bits
    parity_bits = np.dot(parity_check_matrix, bits_with_parity) % 2
    return np.concatenate((bits, parity_bits))

def add_noise(codeword, p):
    noise = np.random.rand(len(codeword)) < p
    noisy_codeword = (codeword + noise) % 2
    return noisy_codeword

def hamming_decode(codeword):
    syndrome = np.dot(parity_check_matrix, codeword) % 2
    error_position = np.sum([2**i for i, bit in enumerate(syndrome) if bit])
    if error_position > 0:
        codeword[error_position - 1] = 1 - codeword[error_position - 1]
    return codeword[:k]

# Gerar dados de treinamento
num_samples = 10000
train_data = np.random.randint(2, size=(num_samples, k))
#print("Training Data is:", train_data[1])
train_codewords = np.array([hamming_encode(bits) for bits in train_data])
#print("\n Training code words are:", train_codewords[1])
train_noisy_codewords = np.array([add_noise(codeword, 0.1) for codeword in train_codewords])
#print("\n Training noisy code words are:", train_noisy_codewords[1])
train_labels = np.array([hamming_decode(codeword) for codeword in train_noisy_codewords])
#print("\n Training labels are:", train_labels[1])
print(train_labels == train_data)

# Construir o modelo
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(n,)),
    tf.keras.layers.Dense(k, activation='sigmoid')
])

# Compilar o modelo
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Treinar o modelo
model.fit(train_noisy_codewords, train_data, epochs=10, batch_size=32)  # Corrigido

# Testar o modelo com um exemplo

test_bits = np.array([0, 1, 0, 1])  # Bits de entrada
test_bits2 = np.array([0, 1, 0, 0])
encoded_bits = hamming_encode(test_bits)  # Codifica os bits usando o código de Hamming
encoded_bits2 = hamming_encode((test_bits2))
print("Encoded bits are: ", encoded_bits)
print("Encoded bits2 are: ", encoded_bits2)

noisy_encoded_bits = add_noise(encoded_bits, 0.1)  # Adiciona ruído aos bits codificados
noisy_encoded_bits2 = add_noise(encoded_bits2, 0.1)  # Adiciona ruído aos bits codificados

print("Noisy bits are:", noisy_encoded_bits)
print("Noisy bits2 are:", noisy_encoded_bits2)
decoded_bits = np.round(model.predict(np.expand_dims(noisy_encoded_bits, axis=0))).astype(int)  # Decodifica os bits usando a rede neural
decoded_bits2 = np.round(model.predict(np.expand_dims(noisy_encoded_bits2, axis=0))).astype(int)  # Decodifica os bits usando a rede neural


print("Bits de entrada:", test_bits)
print("Bits de entrada2:", test_bits2)

print("Bits decodificados pela rede neural:", decoded_bits)
print("Bits2 decodificados pela rede neural:", decoded_bits2)
