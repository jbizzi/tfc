import numpy as np
import matplotlib.pyplot as plt

import HammingCode
import Utils

sample_length = 2 ** 15
noise_rates = np.linspace(0, 1, 101)
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
recalls = []
ber = []
snrs = []
snrs_sem_ruido = []

for noisy_sample in noisy_samples:
    decoded_samples.append(decode_sample(noisy_sample))
    recall, snr, snr_sem_ruido = Utils.calculateRecall(sample, decoded_samples[-1])
    snrs.append(snr)
    snrs_sem_ruido.append(snr_sem_ruido)
    recalls.append(recall)
    ber.append(1 - recall)

#plt.plot(noise_rates, snrs, label='SNR')
#plt.plot(noise_rates, snrs_sem_ruido, label='SNR sem ruído')
plt.figure()
plt.plot(noise_rates, recalls, label='Recall')
plt.plot(noise_rates, ber, label='BER')

plt.xlabel('Noise Rates')
plt.ylabel('Metrics')
plt.legend()
plt.show()


plt.figure()
plt.plot(noise_rates, snrs, label='SNR')
plt.plot(noise_rates, snrs_sem_ruido, label='SNR sem ruído')

plt.xlabel('Noise Rates')
plt.ylabel('Metrics')
plt.xlim(0.05, 1)
plt.ylim(0, 50)
plt.legend()
plt.show()