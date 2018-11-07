# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 16:37:08 2018

@author: mike
"""

import numpy as np

q_cut_off=100
q0=1
q0_amp=1
Hurst=2
N=int(round(q_cut_off/q0))
h, k=range(-1*N,N+1), range(-1*N,N+1)
H,K=np.meshgrid(h,k)
#generate values
mm2=q0_amp**2*((H**2+K**2)/2)**(1-Hurst)
mm2[N,N]=0
pha=2*np.pi*np.random.rand(mm2.shape[0], mm2.shape[1])

mean_mags2=np.zeros((2*N+1,2*N+1))
phases=np.zeros_like(mean_mags2)

mean_mags2[:][N:]=mm2[:][N:]
mean_mags2[:][0:N+1]=np.flipud(mm2[:][N:])
mean_mags=np.sqrt(mean_mags2).flatten()

phases[:][N:]=pha[:][N:]
phases[:][0:N+1]=np.pi*2-np.fliplr(np.flipud(pha[:][N:]))
phases[N,0:N]=np.pi*2-np.flip(phases[N,N+1:])

phases=phases.flatten()

X,Y=np.meshgrid(range(100),range(100))

coords=np.array([X.flatten(), Y.flatten()])

K=np.transpose(H)

qkh=np.transpose(np.array([q0*H.flatten(), q0*K.flatten()]))

Z=np.zeros(X.size,dtype=np.complex64)

for idx in range(len(qkh)):
    Z+=mean_mags[idx]*np.exp(1j*(np.dot(qkh[idx],coords)-phases[idx]))

Z=Z.reshape(X.shape)