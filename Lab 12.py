# -*- coding: utf-8 -*-
################################################
#					                           #
# Jackie Lee 				                   #
# ECE 351 Section 51				           #
# Lab  12 - FInal Project				       #
# Due April 26 2022   			               #
# Filter Design                                #
#                                              #
# Seperated by Comments	                       #
################################################

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import scipy.fftpack
import pandas as pd

#Functions#

def fftfunc(x, fs):
    N = len(x)
    X_fft = scipy.fftpack.fft(x)
    X_fft_shifted = scipy.fftpack.fftshift(X_fft)
    
    freq = np.arange(-N/2, N/2)*fs/N
    
    X_mag = np.abs(X_fft_shifted)/N
    X_phi = np.angle(X_fft_shifted)
    return freq, X_mag, X_phi

def make_stem(ax, x, y, color='k', style='solid', label='', linewidths=2.5, **kwargs):
    ax.axhline(x[0], x[-1],0, color='r')
    ax.vlines(x,0,y,color=color,linestyles=style,label=label,linewidths=linewidths)
    ax.set_ylim([1.05*y.min(), 1.05*y.max()])
    
#Variables#
fs = 1000000 #Sampling Frequency
steps = 1
hz = np.arange(0, 10e6, steps) ## To show frequenices past 100k

#Components#
R = 125.66
L = 100e-3
C = 6.33e-6

#Load Initial Signal Data#

df = pd.read_csv('NoisySignal.csv')

t = df['0'].values
sensor_sig = df['1'].values

#Plot the Initial Signal#

plt.figure(figsize = (10,7))
plt.plot (t, sensor_sig)
plt.grid()
plt.title('Initial Input Signal')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude [V]')
plt.show()


#Transfer Function in Symbolic Form#
num = [R/L, 0]
den = [1, R/L, 1/(L*C)]

#Filtered output#
z, p = sig.bilinear(num, den, 100000) #Up to 100k

outputSignal = sig.lfilter(z, p, sensor_sig)

plt.figure(figsize = (10,7))
plt.plot (t, outputSignal)
plt.grid()
plt.title('Input Signal Through Filter')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude [V]')
plt.show()

#Bode Plots Magnitude and Phase#

sys = sig.TransferFunction(num,den)
w2, mag2, phase2 = sig.bode(sys, hz)

fig, ax = plt.subplots(2)
fig.suptitle("Bode Plot of the Whole Signal in Hz")
plt.tight_layout()
ax[0].semilogx(hz, mag2)
ax[0].set_ylabel("Decibel (dB)")
ax[0].set_title("magnitude")
ax[0].grid()

ax[1].semilogx(hz,phase2)
ax[1].set_xlabel("Frequency (Hz)")
ax[1].set_ylabel("Phase (degrees)")
ax[1].set_title("phase")
ax[1].grid()

fig, ax = plt.subplots(2)
fig.suptitle("Bode Plot of the Low Frequency Vibration Noise")
plt.tight_layout()
ax[0].semilogx(hz, mag2)
ax[0].set_xlim([0,1800])
ax[0].set_ylabel("Decibel (dB)")
ax[0].set_title('magnitude')
ax[0].grid()

ax[1].semilogx(hz,phase2)
ax[1].set_xlim([0,1800])
ax[1].set_xlabel("Frequency (Hz)")
ax[1].set_ylabel("Phase (degrees)")
ax[1].set_title('phase')
ax[1].grid()

fig, ax = plt.subplots(2)
fig.suptitle("Bode Plot of the Switching Amplifer Noise")
plt.tight_layout()
ax[0].semilogx(hz, mag2)
ax[0].set_xlim([2010,100000])
ax[0].set_ylabel("Decibel (dB)")
ax[0].set_title('magnitude')
ax[0].grid()

ax[1].semilogx(hz,phase2)
ax[1].set_xlim([2010,100000])
ax[1].set_xlabel("Frequency (Hz)")
ax[1].set_ylabel("Phase (degrees)")
ax[1].set_title('phase')
ax[1].grid()

fig, ax = plt.subplots(2)
fig.suptitle("Bode Plot of the Position Measurement Information")
plt.tight_layout()
ax[0].semilogx(hz, mag2)
ax[0].set_xlim([1800,2000])
ax[0].set_ylabel("Decibel (dB)")
ax[0].set_title('magnitude')
ax[0].grid()

ax[1].semilogx(hz,phase2)
ax[1].set_xlim([1800,2000])
ax[1].set_xlabel("Frequency (Hz)")
ax[1].set_ylabel("Phase (degrees)")
ax[1].set_title('phase')
ax[1].grid()


#FFT Plots Unfiltered Grouped (First) and Filtered (Second)#

fullFreq, fullXMag, fullXPhi = fftfunc(sensor_sig, fs)
freqFiltered, X_magFiltered, X_phiFiltered = fftfunc(outputSignal, fs)

#Unfiltered#
fig, ax = plt.subplots(figsize=(10,7))
make_stem(ax, fullFreq, fullXMag)
plt.title('Whole Input Signal FFT (Unfiltered)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.grid()
plt.show()

fig, ax = plt.subplots(figsize=(10,7))
ax.set_xlim([0,1800])
make_stem(ax, fullFreq, fullXMag)
plt.title('Low Frequency Vibration Noise FFT (Unfiltered)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.grid()
plt.show()

fig, ax = plt.subplots(figsize=(10,7))
ax.set_xlim([2010, 100000])
make_stem(ax, fullFreq, fullXMag)
plt.title('Switching Amplifer Noise FFT (Unfiltered)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.grid()
plt.show()

fig, ax = plt.subplots(figsize=(10,7))
ax.set_xlim([1800, 2000])
make_stem(ax, fullFreq, fullXMag)
plt.title('Position Measurement Information FFT (Unfiltered)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.grid()
plt.show()

#Filtered#
    
fig, ax = plt.subplots(figsize=(10,7))
make_stem(ax, freqFiltered, X_magFiltered)
plt.title('Whole Input Signal FFT (Filtered)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.grid()
plt.show()

fig, ax = plt.subplots(figsize=(10,7))
ax.set_xlim([0, 1800])
make_stem(ax, freqFiltered, X_magFiltered)
plt.title('Low Frequency Vibration Noise FFT (Filtered)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.grid()
plt.show()

fig, ax = plt.subplots(figsize=(10,7))
ax.set_xlim([2010, 100000])
make_stem(ax, freqFiltered, X_magFiltered)
plt.title('Switching Amplifier Noise FFT (Filtered)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.grid()
plt.show()

fig, ax = plt.subplots(figsize=(10,7))
ax.set_xlim([1800, 2000])
make_stem(ax, freqFiltered, X_magFiltered)
plt.title('Position Measurement Information FFT (Filtered)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.grid()
plt.show()