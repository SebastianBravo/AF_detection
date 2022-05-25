#!/usr/bin/python3
# Example challenge entry

import sys
import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import pan_tompkins as pt
import neurokit2 as nk
import random

# Importar base de datos
# Importar etiquetas 
labels = np.genfromtxt('data/REFERENCE-original.csv',delimiter=',', dtype='str')

# Importar señales con ritmo sinusal normal y AF
normal_idx = np.where(labels[:,1]=='N')[0] # índices ECGs sinusales normales
af_idx = np.where(labels[:,1]=='A')[0] # índices ECGs AF

normal_signals = []
for i in normal_idx:
	normal_signals.append(scipy.io.loadmat(f'data/{labels[i][0]}' + '.mat')['val'].T)

af_signals = []
for i in af_idx:
	af_signals.append(scipy.io.loadmat(f'data/{labels[i][0]}' + '.mat')['val'].T)

# Balancear clases:
# Se tienen 771 muestras de AF y 5154 muestras de Sinusal Normal
# Se temona 771 muestras aleatorias de ritmo Sinusal normal

# random.shuffle(normal_signals)
normal_signals = normal_signals[0:len(af_signals)]

# Unir todos los datos
data = normal_signals + af_signals

# Frecuencia de muestreo
fs = 300


# Extracción de características 
# [Desviación estándar intervalo RR, Relación número ondas p y ondas R, Promedio segmento PR, diferencia RR max y RR min]
features = np.zeros(len(data), )


signal = data[0].reshape(-1,)

# Extracción tiempo picos ondas R, tiempo picos ondas P, inicio intervalo PR y fin intervalo PR 
_, rpeaks = nk.ecg_peaks(signal, sampling_rate=fs)
_, waves_cwt = nk.ecg_delineate(signal, rpeaks, sampling_rate=fs, method="cwt", show=True, show_type='peaks')

# Tiempos ondas R
r_times = rpeaks['ECG_R_Peaks']
r_times = r_times[~np.isnan(r_times)]

# Tiempos ondas P
p_times = np.array(waves_cwt['ECG_P_Peaks'])
p_times = p_times[~np.isnan(p_times)]

# Tiempos inicio intervalo PR
pr_begin_times = np.array(waves_cwt['ECG_P_Onsets'])
pr_begin_times = pr_begin_times[~np.isnan(pr_begin_times)]

# Tiempos fin intervalo PR
pr_end_times = np.array(waves_cwt['ECG_R_Onsets'])
pr_end_times = pr_end_times[~np.isnan(pr_end_times)]


# Intervalos RR
rr_times = (r_times[1:]-r_times[:-1])/fs

# Covaraianza intervalo RR
rr_cov = np.cov(rr_times)

# Desviación estandar intervalo RR
rr_std = np.std(rr_times)

# Relación número ondas p y ondas R
p_r_rel = len(p_times)/len(r_times)

# Diferencia del promedio de los 5 RR max y el promedio de los 5 RR min
rr_diff =(np.mean(np.sort(rr_times)[-5:])) - (np.mean(np.sort(rr_times)[:5]))


# fig, axs = plt.subplots(4)
# fig.suptitle('Etapas pan_tompkins')
# axs[0].plot(real, label='real')
# axs[0].plot(peaks,real[peaks], "x")
# axs[0].plot(lp_signal, label='lp')
# axs[0].plot(hp_signal, label='hp')
# axs[1].plot(deriv_signal, label='der')
# axs[2].plot(sqr_signal, label='sqr')
# axs[3].plot(int_signal, label='int')
# axs[3].plot(peaks,int_signal[peaks], "x")

# axs[0].grid()
# axs[1].grid()
# axs[2].grid()
# axs[3].grid()
# axs[0].set_xlim(0,len(real))
# axs[1].set_xlim(0,len(real))
# axs[2].set_xlim(0,len(real))
# axs[3].set_xlim(0,len(real))

# fig.legend()

plt.plot(signal)
plt.plot(p_times,signal[p_times.astype(int)], "x")
plt.plot(pr_end_times,signal[pr_end_times], "-")
plt.grid()
plt.show()
