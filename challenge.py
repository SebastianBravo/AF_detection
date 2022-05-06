#!/usr/bin/python3
# Example challenge entry

import sys
import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import pan_tompkins as pt

# Importar base de datos
# Importar etiquetas 
labels = np.genfromtxt('data/REFERENCE.csv',delimiter=',', dtype='str')

# Importar señales con ritmo sinusal normal y AF
normal_idx = np.where(labels[:,1]=='N')[0] # índices ECGs sinusales normales
af_idx = np.where(labels[:,1]=='A')[0] # índices ECGs AF
idx = np.concatenate((normal_idx,af_idx), axis=0)

data = []
for i in idx:
	data.append(scipy.io.loadmat(f'data/{labels[i][0]}' + '.mat')['val'].T)

lp_signal = pt.low_pass_filter(data[0])
hp_signal = pt.high_pass_filter(lp_signal)
deriv_signal = pt.derivative_filter(lp_signal,300)
sqr_signal = pt.square_signal(deriv_signal)

# Identificación de picos R usando algoritmo de Pan Tompkins
fs = 300

fig, axs = plt.subplots(3)
fig.suptitle('Etapas pan_tompkins')
axs[0].plot(data[0]/max(abs(data[0])), label='real')
axs[0].plot(lp_signal, label='lp')
axs[0].plot(hp_signal, label='hp')
axs[1].plot(deriv_signal, label='der')
axs[2].plot(sqr_signal, label='sqr')

fig.legend()


plt.show()
