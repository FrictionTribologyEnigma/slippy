# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 17:18:41 2019

@author: 44779
"""

from _johnson_utils import _johnsonsl_old, _johnsonsl
import numpy as np
import scipy.stats
import matplotlib.pyplot as pyplt

step=0.01
a=4
b=5
loc=-4
scale=1

x=np.arange(-5,5,step)

A=_johnsonsl_old(a,b,loc=loc,scale=scale)
B=_johnsonsl(a,b,loc,scale)

rvs=A.rvs(10000)

params=scipy.stats.lognorm.fit(rvs)

print(params)

ya=A.pdf(x)/step
yb=B.pdf(x)/step

pyplt.plot(x,ya)
pyplt.plot(x,yb)