__author__ = 'OksanaS'
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
import pickle

# read python dict back from the file
read_pkl = open('ch1.txt', 'rb')
read = pickle.load(read_pkl)
read_pkl.close()

fs = 256
N = len(read)
t = np.arange(N) / fs

f, Pxx_den = signal.welch(read, fs)
plt.semilogy(f, Pxx_den)
plt.xlabel('frequency')
plt.ylabel('PSD')
plt.show()