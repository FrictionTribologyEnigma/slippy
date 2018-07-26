# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 12:15:36 2018

@author: Michael
"""
import numpy as np

def surf(Z,xmax=0,ymax=0):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        x=range(Z.shape[0])
        y=range(Z.shape[1])
        if xmax:
            x=x/max(x)*xmax
        if ymax:
            y=y/max(y)*ymax
        (X,Y)=np.meshgrid(x,y)
        ax.plot_surface(X,Y,Z)

size=[2,2]
spacing=0.01
nPts=[int(sz/spacing) for sz in size]
profile=np.random.randn(nPts[0],nPts[1])

k=np.arange(-nPts[0]/2,nPts[0]/2)
l=np.arange(-nPts[1]/2,nPts[1]/2)
[K,L]=np.meshgrid(k,l)
sigma=0.01
beta_x=100
beta_y=1000
target_ACF=sigma**2*np.exp(-2.3*np.sqrt((K/beta_x)**2+(L/beta_y)**2))
filter_tf=np.sqrt(np.fft.fft2(target_ACF))

filtered_surface=np.abs(np.fft.ifft2((np.fft.fft2(profile)*filter_tf)))

surf(filtered_surface)

#n=[int(2**(ceil(np.log2(pts*2)))) for pts in nPts]
#target_ACF=sigma**2*np.exp(-2.3*np.sqrt((K/beta_x)**2+(L/beta_y)**2))
#filter_tf=np.sqrt(np.fft.fft2(target_ACF,n))
#
#filtered_surface=np.abs(np.fft.ifft2((np.fft.fft2(profile,n)*filter_tf)))
#
#surf(filtered_surface)
