import numpy as np
from scipy.special import erfc
import matplotlib.pyplot as plt


def calcular_BER_simulada(Eb_dB, tamanho=1e7):

    # Gerar vetor x aleatoriamente com valores 1 ou -1
    x = np.random.choice([1, -1], size=int(tamanho))

    # Variância do ruído
    variancia = 1

    # Gerar ruído gaussiano N com média 0 e variância variancia
    N = np.random.normal(0, np.sqrt(variancia / 2), size=int(tamanho))

    # Valor de Eb
    Eb = 10 ** (Eb_dB / 10)

    # Calcular y = Eb * x + N
    y = np.sqrt(Eb) * x + N

    # Decodificar y como sendo sign(x + N)
    y_decodificado = np.sign(y)

    # Calcular a taxa de erro de bit
    erro_de_bit = np.sum(y_decodificado != x) / tamanho

    return erro_de_bit


# Define os valores de Eb em dB
Eb_dB_values = np.arange(-2, 11, 1)

# Lista para armazenar as taxas de erro de bit simuladas
BER_simulada = []

# Calcular a BER simulada para cada valor de Eb
for Eb_dB in Eb_dB_values:
    BER_simulada.append(calcular_BER_simulada(Eb_dB))
# print(f'BER Simulada para Eb/No de {Eb_dB} dB:', BER_simulada[-1])

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