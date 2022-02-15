# -*- coding: utf-8 -*-
################################################
#					                           #
# Jackie Lee 				                   #
# ECE 351 Section 51				           #
# Lab  4				                       #
# Due February 15 2022 			               #
# System Step Response Using Convolution       #
# Separated by Parts	                       #
################################################

import numpy as np
import matplotlib.pyplot as plt
#####################################Part 1####################################
plt.rcParams.update({'font.size': 14})
steps = 1e-2 
t = np.arange(-10, 10 + steps, steps)
w = .25*2*np.pi #radians per second

def step(t):
    y = np.zeros(t.shape)
    for i in range(len(t)):
            if t[i] < 0:
                y[i] = 0
            else:
                y[i] = 1
    return y


h1 = (np.e**(-2*t))*(step(t)-step(t-3))
h2 = step(t - 2) - step(t - 6)
h3 = (np.cos(w*t))*step(t)

plt.figure(figsize=(12,8))

plt.subplot(3,1,1)
plt.plot(t,h1)
plt.title('Lab 4 Part 1: Signals')

plt.ylabel('h1(t)') 
plt.grid(True)

plt.subplot(3,1,2)
plt.plot(t,h2)
plt.ylabel('h2(t)')
plt.grid(which='both')

plt.subplot(3,1,3)
plt.plot(t,h3)
plt.grid(True)
plt.xlabel('t')
plt.ylabel('h3(t)')
plt.show()

###############################Part 2##########################################

ut = step(t)

newsteps = 1e-2
newt = np.arange(2*t[0], 2*t[len(t)-1]+steps, newsteps)
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
                try:
                    result[i] += F1[j]*F2[i-j+1] #result is the area of the two
                except:
                    print(i,j)
    return result

######Convolution using own function
stepR1 = my_conv(h1,ut)*steps
stepR2 = my_conv(h2,ut)*steps
stepR3 = my_conv(h3,ut)*steps

plt.figure(figsize = (10, 7))
plt.plot(newt, stepR1)
plt.grid()
plt.ylabel('y(t)')
plt.xlabel('t')
plt.axis ([-10, 10, -1, 1])
plt.title('Step Response of Function 1')

plt.figure(figsize = (10, 7))
plt.plot(newt, stepR2)
plt.grid()
plt.ylabel('y(t)')
plt.xlabel('t')
plt.axis ([-10, 10, 0, 5])
plt.title('Step Response of Function 2')

plt.figure(figsize = (10, 7))
plt.plot(newt, stepR3)
plt.grid()
plt.ylabel('y(t)')
plt.xlabel('t')
plt.axis ([-10, 10, -1, 1])
plt.title('Step Response of Function 3')

#####Equations using the Convolution Integral
EqR1 = 0.5*(1-np.e**(-2*newt))*step(newt) - 0.5*(1-np.e**(-2*(newt-3)))*step(newt-3)
EqR2 = (newt-2)*step(newt-2) - (newt-6)*step(newt-6)
EqR3 = (1/(0.5*np.pi))*np.sin(0.5*np.pi*newt)*step(newt)


plt.figure(figsize = (10, 7))
plt.plot(newt, stepR1)
plt.grid()
plt.ylabel('y(t)')
plt.xlabel('t')
plt.axis ([-10, 10, -1, 1])
plt.title('Step Response of Function 1')

plt.figure(figsize = (10, 7))
plt.plot(newt, stepR2)
plt.grid()
plt.ylabel('y(t)')
plt.xlabel('t')
plt.axis ([-10, 10, 0, 5])
plt.title('Step Response of Function 2')

plt.figure(figsize = (10, 7))
plt.plot(newt, stepR3)
plt.grid()
plt.ylabel('y(t)')
plt.xlabel('t')
plt.axis ([-10, 10, -1, 1])
plt.title('Step Response of Function 3')
