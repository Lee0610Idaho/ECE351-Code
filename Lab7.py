# -*- coding: utf-8 -*-
################################################
#					                           #
# Jackie Lee 				                   #
# ECE 351 Section 51				           #
# Lab  7				                       #
# Due March 8 2022   			               #
# Block Diagrams and System Stability          #
#                                              #
# Separated by Parts	                       #
################################################

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

steps = 2e-2

t = np.arange(0, 2, steps)

def stepFunc(t):
    y = np.zeros(t.shape)
    
    for i in range(len(t)):
        if t[i] < 0:
            y[i] = 0
        else:
            y[i] = 1
    return y

###########################Part 1##############################################



#G = (s+9)/((s-8)*(s+2)*(s+4))
#A = (s+4)/((s+3)*(s+1))
#B = (s+14)*(s+12)

zeros, poles, gain = sig.tf2zpk([1,9], sig.convolve([1,-6,-16],[1,4]))
print("G(s):")
print(zeros)
print(poles)
print(gain)

zeros, poles, gain = sig.tf2zpk([1,4], [1,4,3])
print("A(s):")
print(zeros)
print(poles)
print(gain)

roots = np.roots([1,26,168])
print("B(s):")
print(roots) #Only roots
print()

system_open = [1,9],sig.convolve([1,4,3],[1,-6,-16])
#print(sig.convolve([1,4,3],[1,-6,-16]))
t, step = sig.step(system_open)

plt.figure(figsize = (10, 7))
plt.plot(t, step)
plt.grid()
plt.ylabel('y(t)')
plt.xlabel('t')
plt.title('Open Loop Transfer Function Step Response')

print()
###########################Part 2##############################################


numG = [1,9]
denG = sig.convolve([1,-6,-16],[1,4]) #Multiplication of the two
numA = [1,4]
denA = [1,4,3]
numB = [1,26,168]
denB = [1]


numTotal = sig.convolve(numG, numA)
denTotal = sig.convolve(denA, denG) + sig.convolve(sig.convolve(denA, numB), numG)

print(numTotal)
print(denTotal)

transfer_closed = numTotal, denTotal
t, step_closed = sig.step(transfer_closed)

plt.figure(figsize = (10, 7))
plt.plot(t, step_closed)
plt.grid()
plt.ylabel('y(t)')
plt.xlabel('t')
plt.title('Closed Loop Transfer Function Step Response')