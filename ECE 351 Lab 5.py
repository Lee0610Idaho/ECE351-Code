# -*- coding: utf-8 -*-
################################################
#					                           #
# Jackie Lee 				                   #
# ECE 351 Section 51				           #
# Lab  5				                       #
# Due February 22 2022 			               #
# Step & Impulse Response of a RLC Filter      #
# Bandpass                                     #
# Separated by Parts	                       #
################################################

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

##################################Part 1#######################################
steps = 1.2e-5
t = np.arange(0, 0.0012 , steps)

num = [0, 10000, 0]
den = [1, 10000, 370370370]

tout, yout = sig.impulse((num, den), T = t) #Python built in function

#Calculated Impulse Response from Prelab
Imp = 10355.66477*np.e**(-5000*t)*np.sin(18584*t + 1.8326) #radians

plt.figure(figsize=(12,8))

plt.subplot(2,1,1)
plt.plot(t,Imp) #Calculated 
plt.title('Lab 3 Part 1: Impulse Response')

plt.ylabel('Prelab f(t)') 
plt.grid(True)

plt.subplot(2,1,2)
plt.plot(tout,yout) #Python Function
plt.ylabel('sig f(t)')
plt.grid(which='both')

##################################Part 2#######################################
tout, yout = sig.step((num, den), T = t)

plt.figure(figsize=(12,8))
plt.plot(tout, yout)
plt.grid()
plt.ylabel('y(t)')
plt.xlabel('t')
plt.title('Step Response')