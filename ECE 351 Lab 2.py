# -*- coding: utf-8 -*-
################################################
#					                           #
# Jackie Lee 				                   #
# ECE 351 Section 51				           #
# Lab  2				                       #
# Due February 1 2022 			               #
# User Defined Functions Python Code	       #
# Separated by Parts	                       #
################################################

import numpy as np
import matplotlib.pyplot as plt


###############Part 1###########################
plt.rcParams.update({'font.size': 14})
steps = 1e-2 
t = np.arange(0, 10 + steps, steps)

#Create output y(t) for plotting
def func1(t):
    y = np.zeros(t.shape) # Array of zeros
    
    for i in range(len(t)):
            y[i] = np.cos(t[i])
    return y

y = func1(t)

plt.figure(figsize = (10,7))
plt.subplot(2, 1, 1)
plt.plot(t, y)
plt.grid()
plt.ylabel('y(t)')
plt.xlabel('t')
plt.title('Cosine(t)')
#################Part 2##########################
t = np.arange(0, 10 + steps, steps)

#Create output y(t) for plotting
def step(t):
    y = np.zeros(t.shape)
    for i in range(len(t)):
            if t[i] < 0:
                y[i] = 0
            else:
                y[i] = 1
    return y
y = step(t)

plt.figure(figsize = (10,7))
plt.subplot(2, 1, 1)
plt.plot(t, y)
plt.grid()
plt.ylabel('y(t)')
plt.xlabel('t')
plt.title('Step Plot')

t = np.arange(-5, 10 + steps, steps)

def ramp(t):
    y = np.zeros(t.shape)
    for i in range(len(t)):
            if t[i] < 0:
                y[i] = 0
            else:
                y[i] = t[i]
    return y

t = np.arange(-5, 10 + steps, steps)
def func1(t):
    y = np.zeros(t.shape)
    for i in range(len(t)):
        y =  ramp(t) - ramp(t - 3) + 5 * step(t- 3) - 2 * step(t - 6) - 2 * ramp(t - 6) 
    return y

y = func1(t)


plt.figure(figsize = (10,7))
plt.subplot(2, 1, 1)
plt.plot(t, y)
plt.grid()
plt.ylabel('y(t)')
plt.xlabel('t')
plt.title('Fig 2 Plot')

#########################Part 3##################
#(Time Reversal f(-t))
t = np.arange(-10, 5 + steps, steps)
y = func1(-t)

plt.figure(figsize = (10, 7))
plt.plot(t, y)
plt.grid()
plt.ylabel('y(t)')
plt.xlabel('t')
plt.title('Time Reversal Plot')


#(f(t-4))
t = np.arange(0, 15 + steps, steps)
y = func1(t-4)

plt.figure(figsize = (10, 7))
plt.plot(t, y)
plt.grid()
plt.ylabel('y(t)')
plt.xlabel('t')
plt.title('f(t-4) Plot')


#f(-t-4)
t = np.arange(-15, 0 + steps, steps)
y = func1(-t-4)

plt.figure(figsize = (10, 7))
plt.plot(t, y)
plt.grid()
plt.ylabel('y(t)')
plt.xlabel('t')
plt.title('f(-t-4) Plot')


#f(t/2)
t = np.arange(-5, 20 + steps, steps)
y = func1(t/2)

plt.figure(figsize = (10, 7))
plt.plot(t, y)
plt.grid()
plt.ylabel('y(t)')
plt.xlabel('t')
plt.title('f(t/2) Plot')


#f(2t)
t = np.arange(-5, 10 + steps, steps)
y = func1(2*t)

plt.figure(figsize = (10, 7))
plt.plot(t, y)
plt.grid()
plt.ylabel('y(t)')
plt.xlabel('t')
plt.title('f(2t) Plot')

#f'(t)
steps = 1
t = np.arange(-5, 10 + steps, steps)
arr = np.array(func1(t))
dt = np.diff(arr)
plt.figure(figsize = (10, 7))
plt.plot(t, dt[t])
plt.grid()
plt.ylabel('y(t)')
plt.xlabel('t')
plt.title('diff Plot')