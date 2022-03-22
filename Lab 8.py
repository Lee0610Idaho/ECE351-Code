# -*- coding: utf-8 -*-
################################################
#					                           #
# Jackie Lee 				                   #
# ECE 351 Section 51				           #
# Lab  8				                       #
# Due March 22 2022   			               #
# Fourier Series Approximation of a Square Wave#
#                                              #
# Separated by Parts	                       #
################################################


import numpy as np
import matplotlib.pyplot as plt

steps = 2e-2

t = np.arange(0, 20 + 2e-2, steps)

###########################Part 1##############################################

#Initial Summation Values for given N value
s1 = 0
s3 = 0
s15 = 0
s50 = 0
s150 = 0
s1500 = 0


T = 8
w = (2*np.pi)/T
ak = 0
an = 0

## k = 1 ##
print("k=1")
k = 1
bk = 2*((1-np.cos(np.pi * k))/(np.pi * k))
print(bk)
print(ak)

## k = 2 ##
print("k=2")
k = 2
bk = 2*((1-np.cos(np.pi * k))/(np.pi * k))
print(bk)
print(ak)

## k = 3 ##
print("k=3")
k = 3
bk = 2*((1-np.cos(np.pi * k))/(np.pi * k))
print(bk)
print(ak)


for k in range(1,1+1): #+1 due to starting at 1 based on the given series
    bk = 2*((1-np.cos(np.pi * k))/(np.pi * k))
    s1 += bk*np.sin(k*w*t) #svalue for a specified N

for k in range(1,3+1):
    bk = 2*((1-np.cos(np.pi * k))/(np.pi * k))
    s3 += bk*np.sin(k*w*t)
    
for k in range(1,15+1):
    bk = 2*((1-np.cos(np.pi * k))/(np.pi * k))
    s15 += bk*np.sin(k*w*t)
    
for k in range(1,50+1):
    bk = 2*((1-np.cos(np.pi * k))/(np.pi * k))
    s50 += bk*np.sin(k*w*t)

for k in range(1,150+1):
    bk = 2*((1-np.cos(np.pi * k))/(np.pi * k))
    s150 += bk*np.sin(k*w*t)
    
for k in range(1,1500+1):
    bk = 2*((1-np.cos(np.pi * k))/(np.pi * k))
    s1500 += bk*np.sin(k*w*t)
    

#Figure 1 N = 1, 3, 15
plt.subplots(3)
plt.tight_layout()
plt.subplot(3,1,1)
plt.plot(t, s1)
plt.title('N = 1')
plt.grid()

plt.subplot(3,1,2)
plt.plot(t,s3)
plt.title('N = 3')
plt.grid()

plt.subplot(3,1,3)
plt.plot(t,s15)
plt.title('N = 15')
plt.xlabel('t')
plt.grid()

#Figure 2 N = 50, 150, 1500
plt.subplots(3)
plt.tight_layout()
plt.subplot(3,1,1)
plt.plot(t, s50)
plt.title('N = 50')
plt.grid()

plt.subplot(3,1,2)
plt.plot(t,s150)
plt.title('N = 150')
plt.grid()

plt.subplot(3,1,3)
plt.plot(t,s1500)
plt.title('N = 1500')
plt.xlabel('t')
plt.grid()