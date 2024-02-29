import numpy as np
import matplotlib.pyplot as plt
import math
import HammingCode
import Utils

def hamming_distance(sample, noisy_samples):

    decoded_samples = []
    ber_values = []
    snr_values = []

    for noisy_sample in noisy_samples:
        decoded_samples.append(Utils.decode_sample(noisy_sample))
        ber, snr = Utils.calculateRecall(sample, decoded_samples[-1])
        ber_values.append(ber)
        snr_values.append(snr)

    snrs_db = [Utils.dbValueOf(snr) for snr in snr_values]
    plt.figure()
    plt.plot(snrs_db, ber_values)
    plt.xlabel('SNR (dB)')
    plt.xlim(0, 35)
    plt.ylabel('BER')
    plt.yscale("log")
    plt.legend()
    plt.show()