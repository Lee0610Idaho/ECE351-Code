# -*- coding: utf-8 -*-
################################################
#					                           #
# Jackie Lee 				                   #
# ECE 351 Section 51				           #
# Lab  3				                       #
# Due February 8 2022 			               #
# Discrete Convolution               	       #
# Separated by Parts	                       #
################################################

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import math

plt.rcParams.update({'font.size': 14})
steps = 1e-2 
t = np.arange(0, 20 + steps, steps)

##################Part 1########################
def step(t):
    y = np.zeros(t.shape)
    for i in range(len(t)):
            if t[i] < 0:
                y[i] = 0
            else:
                y[i] = 1
    return y

def ramp(t):
    y = np.zeros(t.shape)
    for i in range(len(t)):
            if t[i] < 0:
                y[i] = 0
            else:
                y[i] = t[i]
    return y

def func1(t):
    y = np.zeros(t.shape)
    for i in range(len(t)):
        y =  step(t - 2) - step (t - 9) 
    return y


def func2(t):
    y = np.zeros(t.shape)
    for i in range(len(t)):
        y =  (math.e**-t)*step(t) 
    return y


def func3(t):
    y = np.zeros(t.shape)
    for i in range(len(t)):
        y =  ramp(t - 2) * (step(t - 2) - step(t - 3)) + ramp(4 - t) * (step(t - 3) - step(t - 4))
    return y

y1 = func1(t)
y2 = func2(t)
y3 = func3(t)

plt.figure(figsize=(12,8))

plt.subplot(3,1,1)
plt.plot(t,y1)
plt.title('Lab 3 Part 1: Signals')

plt.ylabel('f1(t)') 
plt.grid(True)

plt.subplot(3,1,2)
plt.plot(t,y2)
plt.ylabel('f2(t)')
plt.grid(which='both')


plt.subplot(3,1,3)
plt.plot(t,y3)
plt.grid(True)
plt.xlabel('t')
plt.ylabel('f3(t)')
plt.show()

####################################Part 2#####################################
newsteps = 0.5e-2
newt = np.arange(0, 20 + 3*newsteps, newsteps)
def my_conv(f1,f2):
    Nf1 = len(f1) #get length of f1
    Nf2 = len(f2) #get length of f2
    F1 = np.append(f1, np.zeros((1, Nf2-1))) #combine the lengths to make them
    F2 = np.append(f2, np.zeros((1, Nf1-1))) #equal by append
    result = np.zeros(F1.shape)
    for i in range(Nf2+Nf1-2): #Range of both functions
        result[i] = 0
        for j in range(Nf1): #Go through the range of the input function
            if(i-j+1>0):
                result[i] += F1[j]*F2[i-j+1] #result is the area of the two
            else:
                #print(i,j)
                pass
    return result

#Convolution using self defined function
f1f2 = my_conv(y1,y2)*steps
f2f3 = my_conv(y2,y3)*steps
f1f3 = my_conv(y1,y3)*steps

#Convolution using built in function
c1 = signal.convolve(y1,y2)*steps
c2 = signal.convolve(y2,y3)*steps
c3 = signal.convolve(y1,y3)*steps

plt.figure(figsize = (10, 7))
plt.plot(newt, f1f2)
plt.grid()
plt.ylabel('y(t)')
plt.xlabel('t')
plt.title('Convolution of f1 and f2 Plot')

plt.figure(figsize = (10, 7))
plt.plot(newt, f2f3)
plt.grid()
plt.ylabel('y(t)')
plt.xlabel('t')
plt.title('Convolution of f2 and f3 Plot')

plt.figure(figsize = (10, 7))
plt.plot(newt, f1f3)
plt.grid()
plt.ylabel('y(t)')
plt.xlabel('t')
plt.title('Convolution of f1 and f3 Plot')

plt.figure(figsize = (10, 7))
plt.plot(newt, c1)
plt.grid()
plt.ylabel('y(t)')
plt.xlabel('t')
plt.title('True Convolution f1, f2 Plot')

plt.figure(figsize = (10, 7))
plt.plot(newt, c2)
plt.grid()
plt.ylabel('y(t)')
plt.xlabel('t')
plt.title('True Convolution f2, f3 Plot')

plt.figure(figsize = (10, 7))
plt.plot(newt, c3)
plt.grid()
plt.ylabel('y(t)')
plt.xlabel('t')
plt.title('True Convolution f1, f3 Plot')

