__author__ = 'OksanaS'
import numpy as np
from edfplus import load_edf
import matplotlib.pyplot as plt
import pickle

struct = load_edf('C_53A_12hr.edf')

data = struct['channels'][5]['data']
label = struct['channels'][5]['label']
unit = struct['channels'][5]['unit']
spr = struct['channels'][5]['spr']

print 'channel data', data
print  'label:', label
print 'unit:', unit
print 'spr:', spr

j = 0
try:
    F4 = [chan["data"] for chan in struct["channels"] if chan["label"] == "F4"][j]
    C4 = [chan["data"] for chan in struct["channels"] if chan["label"] == "C4"][j]
    T4 = [chan["data"] for chan in struct["channels"] if chan["label"] == "T4"][j]
    O2 = [chan["data"] for chan in struct["channels"] if chan["label"] == "O2"][j]
    F3 = [chan["data"] for chan in struct["channels"] if chan["label"] == "F3"][j]
    C3 = [chan["data"] for chan in struct["channels"] if chan["label"] == "C3"][j]
    T3 = [chan["data"] for chan in struct["channels"] if chan["label"] == "T3"][j]
    O1 = [chan["data"] for chan in struct["channels"] if chan["label"] == "O1"][j]
    Cz = [chan["data"] for chan in struct["channels"] if chan["label"] == "Cz"][j]
    channel1 = F4 - C4
    channel2 = C4 - O2
    channel2 = C4 - O1
    channel3 = F3 - C3
    channel4 = C3 - O1
    channel5 = T4 - C4
    channel6 = C4 - Cz
    channel7 = Cz - C3
    channel8 = C3 - T3
except:
        j = j + 1

from numpy import linspace
from pylab import plot, show, title, xlabel, ylabel, subplot
from scipy import fft, arange

def frange(start, stop, step):
    i = start
    while i < stop:
        yield i
        i += step

t = np.hstack(frange(0, 3601, 0.00390625))
#NOTE: t = range(0, 3601, 1) - for 1 spr

def plotSpectrum(y,Fs):
 """
 Plots a Single-Sided Amplitude Spectrum of y(t)
 """
 n = len(y) # length of the signal
 k = arange(n)
 T = n/Fs
 frq = k/T # two sides frequency range
 frq = frq[range(n/2)] # one side frequency range

 Y = fft(y)/n # fft computing and normalization
 Y = Y[range(n/2)]

 plot(frq,abs(Y),'r') # plotting the spectrum


Fs = 256.0;  # sampling rate
Ts = 1.0/Fs; # sampling interval
y = channel8


#Periodogram
from scipy import signal

f, Pxx_den = signal.welch(channel8, Fs)
plt.semilogy(f, Pxx_den)
plt.show()


subplot(3,1,1)
plot(t,y)
xlabel('Time')
ylabel('Amplitude')
subplot(3,1,2)
plotSpectrum(y,Fs)
xlabel('Frequency')
subplot(3,1,3)
plt.semilogy(f, Pxx_den)
xlabel('Frequency')
ylabel('PSD')
show()

print struct['labels']













