import numpy as np
import matplotlib.pyplot as plt
import math
import HammingCode
import Utils

sample_length = 2 ** 10
noise_rates = np.linspace(0, 1, 1001)
sample = Utils.generate_random_string(sample_length)


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


encoded_sample = encode_sample(sample)

noisy_samples = []
for rate in noise_rates:
    noisy_samples.append(Utils.noiseString(rate, encoded_sample))

decoded_samples = []
ber_values = []
snr_values = []

for noisy_sample in noisy_samples:
    decoded_samples.append(decode_sample(noisy_sample))
    ber, snr = Utils.calculateRecall(sample, decoded_samples[-1])
    ber_values.append(ber)
    snr_values.append(snr)

print(noise_rates)
snrs_db = [Utils.dbValueOf(snr) for snr in snr_values]
plt.figure()
plt.plot(snrs_db, ber_values)
plt.xlabel('SNR (dB)')
plt.xlim(0, 35)
plt.ylabel('BER')
plt.yscale("log")
plt.legend()
plt.show()