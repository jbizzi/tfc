import numpy as np
import matplotlib.pyplot as plt
import math
import HammingCode
import Utils

from scipy.special import erfc

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



Eb_dB_values = np.arange(-2, 11, 1)
BER_simulada = noise_rates*sample_length

# Calcular a BER simulada para cada valor de Eb
for Eb_dB in Eb_dB_values:
    np.append(BER_simulada, Utils.calcular_BER_simulada(Eb_dB))

# Converter a lista BER_simulada em array numpy
BER_simulada = np.array(BER_simulada)

# Calcular a BER teórica
Eb_values = 10 ** (Eb_dB_values / 10)


BER_teorica = 0.5 * erfc(np.sqrt(Eb_values))

# Imprimir as taxas de erro de bit teóricas
# for i, Eb_dB in enumerate(Eb_dB_values):
#    print(f'BER Teórica para Eb/No de {Eb_dB} dB:', BER_teorica[i])

# Plotar as taxas de erro de bit simuladas e teóricas
plt.figure(figsize=(10, 6))
plt.semilogy(Eb_dB_values, BER_simulada, marker='o', label='BER Simulada')
plt.semilogy(Eb_dB_values, BER_teorica, linestyle='--', label='BER Teórica')
plt.xlabel('Eb/No (dB)')
plt.ylabel('BER')
plt.grid(True, which='both')
plt.legend()
plt.title('BER Simulada vs BER Teórica')
plt.show()

'''

snrs_db = [Utils.dbValueOf(snr) for snr in snr_values]
plt.figure()
plt.plot(snrs_db, ber_values)
plt.xlabel('SNR (dB)')
plt.xlim(0, 35)
plt.ylabel('BER')
plt.yscale("log")
plt.legend()
plt.show()'''