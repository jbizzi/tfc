import random
import numpy as np


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
    recall = equals/len(original)
    snr = len(original)/(incorrect_bits if incorrect_bits != 0 else 1 )
    snr_without_noise = (len(original) - incorrect_bits)/incorrect_bits if incorrect_bits != 0 else 1
    return recall,snr, snr_without_noise

