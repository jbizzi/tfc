import random

import numpy as np
import Utils

# Generator matrix G
G = np.array([
    [1, 1, 1, 0, 0, 0, 0],
    [1, 0, 0, 1, 1, 0, 0],
    [0, 1, 0, 1, 0, 1, 0],
    [1, 1, 0, 1, 0, 0, 1]
])

# Parity check matrix H
H = np.array([
    [1, 0, 1, 0, 1, 0, 1],
    [0, 1, 1, 0, 0, 1, 1],
    [0, 0, 0, 1, 1, 1, 1]
])

messages = [
    [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 0, 1, 1],
    [0, 1, 0, 0], [0, 1, 0, 1], [0, 1, 1, 0], [0, 1, 1, 1],
    [1, 0, 0, 0], [1, 0, 0, 1], [1, 0, 1, 0], [1, 0, 1, 1],
    [1, 1, 0, 0], [1, 1, 0, 1], [1, 1, 1, 0], [1, 1, 1, 1]
]

data_bits = [2, 4, 5, 6]

def add_noise(data, index):
    data = list(data)
    data[index] = str(int(data[index]) ^ 1)  # Flipping the bit
    return data

# First four bits are data, fifth, sixth and seventh are parity checks
def encode(data):
    return Utils.toBooleanList(np.dot(np.array([int(bit) for bit in data]), G) % 2)

# Decode function
def decode(received_code):
    received_code = np.array([int(bit) for bit in received_code])
    syndrome = np.dot(received_code, H.T) % 2
    # Check if syndrome is non-zero
    if np.any(syndrome):
        # Determine the position of the erroneous bit
        error_position = np.sum(syndrome * [1, 2, 4]) - 1

        # Correct the bit at the determined position
        received_code[error_position] = 1 - received_code[error_position]

    return received_code[[2, 4, 5, 6]]

def decode_sample(data):
    decoded_data = []
    for i in range(0, len(data) - 6, 7):
        decoded_data.extend(decode(data[i:i + 7]).T)
    return decoded_data

def encode_sample(data):
    encoded_data = []
    for i in range(0, len(data) - 3, 4):
        encoded_data.extend(encode(data[i:i + 4]))
    return encoded_data
