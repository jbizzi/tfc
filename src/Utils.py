import random
import numpy as np
import math

from src import HammingCode


def encode_sample(data):
    encoded_data = []
    for i in range(0, len(data) - 3, 4):
        encoded_data.extend(HammingCode.encode(data[i:i + 4]))
    return encoded_data


def decode_sample(data):
    decoded_data = []
    for i in range(0, len(data) - 6, 7):
        decoded_data.extend(HammingCode.decode(data[i:i + 7]).T)
    return decoded_data


def generate_random_string(length):

    # Define the characters to choose fromstring.digits
    characters = ['0', '1']

    # Generate a random string of specified length
    random_string = ''.join(random.choice(characters) for _ in range(length))

    return random_string

def checkEquals(received, expected):
    for index in range(0, len(received)):
        if int(received[index]) != int(expected[index]):
            return False
    return True

def toInt(list):
    return [int(string_value) for string_value in list]

def noiseString(rate, word):
    word = np.array([int(bit) for bit in word])
    bits_to_flip = round(len(word) * rate, 0)

    noise_positions = np.array([random.randint(0, len(word) - 1) for i in range(0, int(bits_to_flip))])

    for each in noise_positions:
        word[each] = str(int(word[each]) ^ 1)

    return word

def calculateRecall(original, decoded):
    equals = 0
    for i in range(0, len(original)):
        if (int(original[i]) == int(decoded[i])):
            equals += 1
    incorrect_bits = (len(original) - equals)
    ber = incorrect_bits/len(original)
    snr = equals/(incorrect_bits if incorrect_bits != 0 else 1)
    return ber,snr

def dbValueOf(value):
    return 10 * math.log10(value)

def generateDataSet(length):

    # Basic setup
    noise_rates = np.linspace(0, 1, 1001)
    sample = generate_random_string(length)

    # Encode
    encoded_sample = encode_sample(sample)

    # Insert Noise
    noisy_samples = []
    for rate in noise_rates:
        noisy_samples.append(noiseString(rate, encoded_sample))
    return noisy_samples, encoded_sample, sample, noise_rates