import numpy as np
import scipy.integrate as integrate
import scipy.special as special
from matplotlib import pyplot as plt

def lazynan(function, params):
    value=function(params)
    if np.isnan(value):
        value=0
    #if value>1E100:
        #value=value/(value-1E100)
    return value   

E=1
V=0.3
log_roh=np.linspace(-4,2,100)
rohs=np.exp(log_roh)
hors=1/rohs
S = lambda hor, w, v : (((3-4*v)*np.sinh(2*w*hor)-2*w*hor)/
                        (5+2*(w*hor)**2-4*v*(3-2*v)+(3-4*v)*np.cosh(2*w*hor)))

B=4*V*(3-2*V)
C=5-B
A=3-4*V

S2=lambda hor, w : (A*(np.exp(4*w*hor)-1)-2*w*hor*2*np.exp(2*w*hor))/(A*(np.exp(2*2*w*hor)+1)+(2*w*hor)**2*np.exp(2*w*hor)+C*(np.exp(2*w*hor)))

theta=[]


for hor in hors:
    s=lambda w : S2(hor,(w-1))*special.jv(0,(w-1))
    
    no_nan=lambda w : lazynan(s,w)
    
    theta.append(integrate.quad(no_nan,1,np.inf,limit=1000000,epsrel=1e-20))

plt.plot(log_roh,theta)