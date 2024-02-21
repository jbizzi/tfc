import random

import numpy as np

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
    return np.dot(data, G) % 2

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

def checkEquals(received, expected):
    for index in [0, len(received)]:
        if received[index] != expected[index]:
            return False
    return True



encoded_messages = np.array([encode(message) for message in messages])
noise_messages = np.array([add_noise(encoded_message, data_bits[random.randint(0, 3)]) for encoded_message in encoded_messages])
print(noise_messages)
decoded_messages = np.array([decode(noisy_message) for noisy_message in encoded_messages])

for i in range(len(encoded_messages)):
    print("DATA: ", messages[i],"| ENCODED: ", encoded_messages[i], " | NOISY: ", noise_messages[i], " | FIXED IS: ", decoded_messages[i])
#print(checkEquals(decoded_messages, messages))
