# -*- coding: utf-8 -*-
################################################
#					                           #
# Jackie Lee 				                   #
# ECE 351 Section 51				           #
# Lab  11				                       #
# Due April 12 2022   			               #
# Z - Transform Operations                     #
#                                              #
#                   	                       #
################################################

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig


num = [2, -40, 0]
den = [1, -10, 16]

r, p, k = sig.residuez(num, den)

print(r)
print(p)

def zplane (b , a , filename = None ) :
 """ Plot the complex z- plane given a transfer function """

 import numpy as np
 import matplotlib . pyplot as plt
 from matplotlib import patches

 # get a figure / plot
 ax = plt . subplot (1 , 1 , 1)

 # create the unit circle
 uc = patches . Circle ((0 ,0) , radius =1 , fill = False , color ="black", ls ="dashed")
 ax . add_patch ( uc )

 # the coefficients are less than 1 , normalize the coefficients
 if np . max( b ) > 1:
     kn = np . max( b )
     b = np . array ( b ) / float ( kn )
 else :

     kn = 1

 if np . max( a ) > 1:
     kd = np . max( a )
     a = np . array ( a ) / float ( kd )
 else :
     kd = 1

 # get the poles and zeros
 p = np . roots ( a )
 z = np . roots ( b )
 k = kn / float ( kd )

 # plot the zeros and set marker properties
 t1 = plt . plot ( z . real , z . imag , "o", ms =10 , label ="Zeros")
 plt . setp ( t1 , markersize =10.0 , markeredgewidth =1.0)

 # plot the poles and set marker properties
 t2 = plt . plot ( p . real , p . imag , "x", ms =10 , label ="Poles")
 plt . setp ( t2 , markersize =12.0 , markeredgewidth =3.0)

 ax . spines ["left"]. set_position ("center")
 ax . spines ["bottom"]. set_position ("center")
 ax . spines ["right"]. set_visible ( False )
 ax . spines ["top"]. set_visible ( False )

 plt . legend ()

 # set the ticks

 # r = 1.5; plt. axis ( ’ scaled ’); plt. axis ([ -r, r, -r, r])
 # ticks = [ -1 , -.5 , .5 , 1]; plt. xticks ( ticks ); plt. yticks ( ticks )

 if filename is None :
     plt . show ()
 else :
     plt . savefig ( filename )

 return z , p , k

z, p, k = zplane(num, den)

w, h = sig.freqz(num, den, whole=True)

plt.subplots(2)
plt.tight_layout()
plt.subplot(2,1,1)
plt.plot(w, abs(h))
plt.title('magnitude')
plt.grid()

plt.subplot(2,1,2)
plt.plot(w, np.angle(h))
plt.title('phase')
plt.xlabel('Frequency [rad/sample]')
plt.grid()