import sys
import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import pan_tompkins as pt
import neurokit2 as nk
import random
from scipy.signal import find_peaks

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
# [Covarianza intervalo RR, Desviación estándar intervalo RR, diferencia RR max y RR min, Relación número ondas p y ondas R]
X = np.zeros((len(data),4))

for i in range(len(data)):
	signal = data[i].reshape(-1,)

	# Extracción tiempo picos ondas R, tiempo picos ondas P, inicio intervalo PR y fin intervalo PR 
	_, rpeaks = nk.ecg_peaks(signal, sampling_rate=fs)

	# Tiempos ondas R
	r_times = rpeaks['ECG_R_Peaks']
	r_times = r_times[~np.isnan(r_times)]

	# Intervalos RR
	rr_times = (r_times[1:]-r_times[:-1])

	# Covaraianza intervalo RR
	rr_cov = np.cov(rr_times/fs)

	# Desviación estandar intervalo RR
	rr_std = np.std(rr_times/fs)

	# Diferencia del promedio de los 5 RR max y el promedio de los 5 RR min
	rr_diff =(np.mean(np.sort(rr_times/fs)[-5:])) - (np.mean(np.sort(rr_times/fs)[:5]))

	# Tiempos ondas P
	p_times = np.zeros(len(rr_times),)

	for j in range(len(r_times)-1):
		inicio = int(r_times[j+1] - 0.3*rr_times[j])
		fin = int(r_times[j+1] - 0.1*rr_times[j])

		signal_p = np.zeros(len(signal),)
		signal_p[inicio:fin] = signal[inicio:fin]

		p_times[j] = np.argmax(signal_p) if signal[np.argmax(signal_p)]>3*np.mean(signal) else None

	p_times = p_times[~np.isnan(p_times)]

	# Relación número ondas p y ondas R
	p_r_rel = len(p_times)/len(r_times)

	X[i] = np.array([rr_cov,rr_std,rr_diff,p_r_rel])

# Etiquetas 0 == Ritmo Sinusal Normal, 1 == AF
y = np.concatenate((np.zeros(len(normal_signals),), np.ones(len(af_signals))),axis=0)

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
plt.plot(r_times,signal[r_times.astype(int)], "x")
# plt.ylim(min(signal),np.mean(signal))
# plt.plot(pr_end_times,signal[pr_end_times], "-")
# plt.grid()
# plt.show()
