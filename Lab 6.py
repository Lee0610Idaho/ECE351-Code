# -*- coding: utf-8 -*-
################################################
#					                           #
# Jackie Lee 				                   #
# ECE 351 Section 51				           #
# Lab  6				                       #
# Due March 1 2022   			               #
# Partial Fraction Expansion                   #
#                                              #
# Separated by Parts	                       #
################################################

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

###########################Part 1##############################################
steps = 2e-2
t = np.arange(0, 2, steps) #plot 0 to 2 seconds
def step(t):
    y = np.zeros(t.shape)
    for i in range(len(t)):
            if t[i] < 0:
                y[i] = 0
            else:
                y[i] = 1
    return y

def delta(t): #Not necessary ut thought I needed it
    y = np.zeros(t.shape)
    for i in range(len(t)):
            if t[i] == 0:
                y[i] = 100
            else:
                y[i] = 0
    return y

stepresponse = (np.e**(-6*t)-0.5*np.e**(-4*t)+0.5)*step(t)

num = [1, 6, 12]
den = [1, 10, 24]

tout, yout = sig.step((num, den), T = t) #Step Response

plt.figure(figsize = (10, 7))
plt.subplot(2,1,1)
plt.plot(t,stepresponse) #Calculated 
plt.title('Lab 6 Part 1: Step Response')

plt.ylabel('Prelab f(t)') 
plt.grid(True)

plt.subplot(2,1,2)
plt.plot(tout,yout) #Python Function
plt.ylabel('step f(t)')
plt.xlabel('t')
plt.grid(which='both')

R, P, K = sig.residue([1,6,12], [1,10,24,0])

print('Part 1')
print(R)
print(P)
print(K)


##############################Part 2###########################################
newt = np.arange(0, 4.5, steps)
R2, P2, K2 = sig.residue([0,0,0,0,0,0,25250], [1,18,218,2036,9085,25250,0])

print('Part 2')
print(R2)
print(P2)
print(K2)


tdr = 0 #Initial Value as we're doing a sum from a loop
for i in range(len(R2)):
    w = np.imag(P2[i]) #return imaginary part 
    kabs = np.abs(R2[i]) #returns the magnitude of complex number
    krad = np.angle(R2[i]) #returns radians for k
    alpha = np.real(P2[i]) #only need real part of complex number
    eAlphat = np.e**(alpha*newt) ##e to the alpha t
    cos = np.cos((w * newt) + krad )
    tdr += kabs*eAlphat*cos*step(newt)
    
TempTrans2 = ([25250], [1,18,218,2036,9085,25250]) #System for step function

t, step2 = sig.step(TempTrans2)

plt.figure(figsize = (10, 7))
plt.subplot(2,1,1)
plt.plot(newt,tdr) #Cosine Method
plt.title('Lab 6 Part 2: Time Domain Response')

plt.ylabel('Cosine Method f(t)') 
plt.grid(True)

plt.subplot(2,1,2)
plt.plot(t, step2) #Python Function
plt.axis ([0.0, 4.5, 0.0, 1.2])
plt.ylabel('step f(t)')
plt.xlabel('t')
plt.grid(which='both')

