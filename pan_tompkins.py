import numpy as np

def low_pass_filter(signal):
	# y(nT) = 2y(nT - T) - y(nT - 2 T) + x(nT) - 2x(nT- 6T) + x(nT- 12T) 

	# Inicializar salida en 0:
	y = np.zeros((len(signal),1))

	# Filtro pasa bajos:
	for i in range(len(signal)):
		y[i] = signal[i] # x(nT)

		if (i >= 1):
			y[i] += 2*y[i-1] # + 2y(nT - T)

		if (i >= 2):
			y[i] -= y[i-2] # - y(nT - 2T)

		if (i >= 6):
			y[i] -= 2*signal[i-6] # - 2x(nT - 6T)

		if (i >= 12):
			y[i] += signal[i-12]  # + x(nT - 12T)

	return y/max(abs(y)) # Salida normalizada

def high_pass_filter(signal):	
	# y(nT) = 32x(nT - 16 T) - [y(nT - T) + x(nT) - x(nT - 32 T)]
	
	# Inicializar salida en 0:
	y = np.zeros((len(signal),1))

	# Filtro pasa altos:
	for i in range(len(signal)):
		y[i] = -signal[i] # - x(nT)

		if (i >= 1):
			y[i] -= y[i-1] # - y(nT - T)

		if (i >= 16):
			y[i] += 32*signal[i-16] # + 32x(nT - 16T)

		if (i >= 32):
			y[i] += signal[i-32] # + x(nT - 32T)
	
	return y/max(abs(y)) # Salida normalizada

def derivative_filter(signal,fs):
	# y(nT) = (1/8 T) [-x(nT - 2 T) - 2x(nT - T) + 2x(nT + T) + x(nT+ 2T)]

	# Inicializar salida en 0:
	y = np.zeros((len(signal),1))

	# Filtro pasa altos:
	for i in range(len(signal)):
		if (i >= 1):
			y[i] -= 2*signal[i-1] # - 2x(nT - T)

		if (i >= 2):
			y[i] -= signal[i-2] # -x(nT - 2 T)

		if (i <= len(signal)-2):
			y[i] += 2*signal[i+1] # + 2x(nT + T)

		if (i <= len(signal)-3):
			y[i] += signal[i+2] # + x(nT+ 2T)

	y = fs*y/8 # (1/8 T)

	return y

def square_signal(signal):
	y = np.square(signal)

	return y

