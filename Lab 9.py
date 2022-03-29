# -*- coding: utf-8 -*-
################################################
#					                           #
# Jackie Lee 				                   #
# ECE 351 Section 51				           #
# Lab 9				                           #
# Due March 29 2022   			               #
# Fast Fourier Transform                       #
#                                              #
# Separated by Tasks	                       #
################################################

import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack

steps = 1e-2

t = np.arange(0, 2, steps)


###########################Task 1##############################################

fs = 100

def fftfunc(x, fs):
    N = len(x)
    X_fft = scipy.fftpack.fft(x)
    X_fft_shifted = scipy.fftpack.fftshift(X_fft)
    
    freq = np.arange(-N/2, N/2)*fs/N
    
    X_mag = np.abs(X_fft_shifted)/N
    X_phi = np.angle(X_fft_shifted)
    return freq, X_mag, X_phi
    
x = np.cos(np.pi * 2 * t)  
freq, X_mag, X_phi = fftfunc(x, fs)

fig = plt.figure()

fig1 = plt.subplot2grid((3,2), (0,0), colspan=2, rowspan=1)
fig1.plot(t,x)
fig1.title.set_text("Task 1")
fig1.grid()

fig2 = plt.subplot2grid((3,2), (1,0), colspan=1, rowspan=1)
fig2.stem(freq, X_mag)
fig2.grid()

fig3 = plt.subplot2grid((3,2), (1,1), colspan=1, rowspan=1)
fig3.set_xlim([-2,2])
fig3.stem(freq, X_mag)
fig3.grid()

fig4 = plt.subplot2grid((3,2), (2,0), colspan=1, rowspan=1)
fig4.stem(freq, X_phi)
fig4.grid()

fig5 = plt.subplot2grid((3,2), (2,1), colspan=1, rowspan=1)
fig5.stem(freq, X_phi)
fig5.set_xlim([-2,2])
fig5.grid()

fig.tight_layout()

################################Task 2#########################################

x = 5*np.sin(2*np.pi*t)

freq, X_mag, X_phi = fftfunc(x, fs)

fig = plt.figure()

fig1 = plt.subplot2grid((3,2), (0,0), colspan=2, rowspan=1)
fig1.plot(t,x)
fig1.title.set_text("Task 2")
fig1.grid()

fig2 = plt.subplot2grid((3,2), (1,0), colspan=1, rowspan=1)
fig2.stem(freq, X_mag)
fig2.grid()

fig3 = plt.subplot2grid((3,2), (1,1), colspan=1, rowspan=1)
fig3.set_xlim([-2,2])
fig3.stem(freq, X_mag)
fig3.grid()

fig4 = plt.subplot2grid((3,2), (2,0), colspan=1, rowspan=1)
fig4.stem(freq, X_phi)
fig4.grid()

fig5 = plt.subplot2grid((3,2), (2,1), colspan=1, rowspan=1)
fig5.stem(freq, X_phi)
fig5.set_xlim([-2,2])
fig5.grid()

fig.tight_layout()

################################Task 3#########################################

x = 2*np.cos((2*np.pi*2*t) - 2) + np.sin((2*np.pi*6*t)+3)**2

freq, X_mag, X_phi = fftfunc(x, fs)

fig = plt.figure()

fig1 = plt.subplot2grid((3,2), (0,0), colspan=2, rowspan=1)
fig1.plot(t,x)
fig1.title.set_text("Task 3")
fig1.grid()

fig2 = plt.subplot2grid((3,2), (1,0), colspan=1, rowspan=1)
fig2.stem(freq, X_mag)
fig2.grid()

fig3 = plt.subplot2grid((3,2), (1,1), colspan=1, rowspan=1)
fig3.set_xlim([-15,15])
fig3.stem(freq, X_mag)
fig3.grid()

fig4 = plt.subplot2grid((3,2), (2,0), colspan=1, rowspan=1)
fig4.stem(freq, X_phi)
fig4.grid()

fig5 = plt.subplot2grid((3,2), (2,1), colspan=1, rowspan=1)
fig5.stem(freq, X_phi)
fig5.set_xlim([-15,15])
fig5.grid()

fig.tight_layout()

################################Task 4#########################################

def fftfunc2(x, fs):
    N = len(x)
    X_fft = scipy.fftpack.fft(x)
    X_fft_shifted = scipy.fftpack.fftshift(X_fft)
    
    freq = np.arange(-N/2, N/2)*fs/N
    
    X_mag = np.abs(X_fft_shifted)/N
    X_phi = np.angle(X_fft_shifted)
    for i in range(len(X_mag)):
        if X_mag[i] < 1e-10:
            X_phi[i] = 0
            
    return freq, X_mag, X_phi

##### Rerun Task 1  #####

x = np.cos(np.pi * 2 * t)   
    
freq, X_mag, X_phi = fftfunc2(x, fs)

fig = plt.figure()

fig1 = plt.subplot2grid((3,2), (0,0), colspan=2, rowspan=1)
fig1.plot(t,x)
fig1.title.set_text("Task 1")
fig1.grid()

fig2 = plt.subplot2grid((3,2), (1,0), colspan=1, rowspan=1)
fig2.stem(freq, X_mag)
fig2.grid()

fig3 = plt.subplot2grid((3,2), (1,1), colspan=1, rowspan=1)
fig3.set_xlim([-2,2])
fig3.stem(freq, X_mag)
fig3.grid()

fig4 = plt.subplot2grid((3,2), (2,0), colspan=1, rowspan=1)
fig4.stem(freq, X_phi)
fig4.grid()

fig5 = plt.subplot2grid((3,2), (2,1), colspan=1, rowspan=1)
fig5.stem(freq, X_phi)
fig5.set_xlim([-2,2])
fig5.grid()

fig.tight_layout()

##### Rerun Task 2  #####

x = 5*np.sin(2*np.pi*t)

freq, X_mag, X_phi = fftfunc2(x, fs)

fig = plt.figure()

fig1 = plt.subplot2grid((3,2), (0,0), colspan=2, rowspan=1)
fig1.plot(t,x)
fig1.title.set_text("Task 2")
fig1.grid()

fig2 = plt.subplot2grid((3,2), (1,0), colspan=1, rowspan=1)
fig2.stem(freq, X_mag)
fig2.grid()

fig3 = plt.subplot2grid((3,2), (1,1), colspan=1, rowspan=1)
fig3.set_xlim([-2,2])
fig3.stem(freq, X_mag)
fig3.grid()

fig4 = plt.subplot2grid((3,2), (2,0), colspan=1, rowspan=1)
fig4.stem(freq, X_phi)
fig4.grid()

fig5 = plt.subplot2grid((3,2), (2,1), colspan=1, rowspan=1)
fig5.stem(freq, X_phi)
fig5.set_xlim([-2,2])
fig5.grid()

fig.tight_layout()

##### Rerun Task 3  #####

x = 2*np.cos((2*np.pi*2*t) - 2) + np.sin((2*np.pi*6*t)+3)**2

freq, X_mag, X_phi = fftfunc2(x, fs)

fig = plt.figure()

fig1 = plt.subplot2grid((3,2), (0,0), colspan=2, rowspan=1)
fig1.plot(t,x)
fig1.title.set_text("Task 3")
fig1.grid()

fig2 = plt.subplot2grid((3,2), (1,0), colspan=1, rowspan=1)
fig2.stem(freq, X_mag)
fig2.grid()

fig3 = plt.subplot2grid((3,2), (1,1), colspan=1, rowspan=1)
fig3.set_xlim([-15,15])
fig3.stem(freq, X_mag)
fig3.grid()

fig4 = plt.subplot2grid((3,2), (2,0), colspan=1, rowspan=1)
fig4.stem(freq, X_phi)
fig4.grid()

fig5 = plt.subplot2grid((3,2), (2,1), colspan=1, rowspan=1)
fig5.stem(freq, X_phi)
fig5.set_xlim([-15,15])
fig5.grid()

fig.tight_layout()


################################Task 5#########################################

t = np.arange(0, 16, steps)

T = 8 
w = (2*np.pi)/T
s15 = 0 ##N = 15

for k in range(1,15+1):
    bk = 2*((1-np.cos(np.pi * k))/(np.pi * k))
    s15 += bk*np.sin(k*w*t)

freq, X_mag, X_phi = fftfunc2(s15, fs)

fig = plt.figure()

fig1 = plt.subplot2grid((3,2), (0,0), colspan=2, rowspan=1)
fig1.plot(t,s15)
fig1.title.set_text("Task 5")
fig1.grid()

fig2 = plt.subplot2grid((3,2), (1,0), colspan=1, rowspan=1)
fig2.stem(freq, X_mag)
fig2.grid()

fig3 = plt.subplot2grid((3,2), (1,1), colspan=1, rowspan=1)
fig3.set_xlim([-2,2])
fig3.stem(freq, X_mag)
fig3.grid()

fig4 = plt.subplot2grid((3,2), (2,0), colspan=1, rowspan=1)
fig4.stem(freq, X_phi)
fig4.grid()

fig5 = plt.subplot2grid((3,2), (2,1), colspan=1, rowspan=1)
fig5.stem(freq, X_phi)
fig5.set_xlim([-2,2])
fig5.grid()

fig.tight_layout()