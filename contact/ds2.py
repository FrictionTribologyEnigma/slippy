# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 17:08:14 2019

@author: Michael
"""
import numpy as np
import scipy.special as sp
import matplotlib.pyplot as plt

B=4*V*(3-2*V)
C=5-B
A=3-4*V

I= lambda w: np.pi*w/2*sp.struve(0, w)*sp.jv(1,w)+w/2*(2-np.pi*sp.struve(1,w))*sp.jv(0,w)
dsdw= lambda

logx=np.linspace(0, 20, 500)
x=np.exp(logx)
y=I(x)
plt.plot(logx,y)