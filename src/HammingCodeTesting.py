import numpy as np

import HammingCode
import Utils

noise_rates = np.linspace(0, 1, 11)
sample = Utils.generate_random_string(32)
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

# decoded data is only adding first?

encoded_sample = encode_sample(sample)

noisy_samples = []
for rate in noise_rates:
    noisy_samples.append(Utils.noiseString(rate, encoded_sample))


decoded_samples = []
eficiency = []

for noisy_sample in noisy_samples:
    decoded_samples.append(decode_sample(noisy_sample))
    print(sample, decoded_samples[-1])
    eficiency.append(Utils.caculateEquality(sample, decoded_samples[-1]))

print(eficiency)


