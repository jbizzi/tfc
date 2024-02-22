import numpy as np

import HammingCode
import Utils

noise_rates = np.linspace(0, 1, 11)
sample = Utils.generate_random_string(1024)
def encode_sample(data):
    encoded_data = []
    for i in range(0, len(data), 4):
        encoded_data.extend(HammingCode.encode(data[i:i + 4]))
    return encoded_data

def decode_sample(data):
    decoded_data = []
    for i in range(0, len(data), 7):
        decoded_data.extend(HammingCode.decode(data[i:i + 7]))
    return decoded_data

encoded_sample = encode_sample(sample)
decoded_sample = decode_sample(encoded_sample)

print(Utils.checkEquals(list(sample), decoded_sample))



