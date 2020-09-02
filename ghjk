import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import sympy as sm
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

g=pd.read_excel("C:/Users/dell/Desktop/lol.xlsx")
ab=g['体重'].values
cd=g['身高'].values
ef=g['肺活量'].values
l=len(ab)

ab=ab.reshape((len(ab),1))
cd=cd.reshape((len(cd),1))
ef=ef.reshape((len(ef),1))
m=np.hstack((ab**2,cd))
ones=np.ones((l,1))
m=np.hstack((ones,m))
theta=np.dot(np.dot(np.linalg.inv(np.dot(m.T,m)),m.T),ef)

fig=plt.figure()
ax2 = Axes3D(fig)
x0=ones.reshape(1,l)
x1=ab.reshape(1,l)
x4=x1**2+x0
x2=cd.reshape(1,l)
x1,x2=np.meshgrid(x1**2,x2**2)
x3=x0*theta[0]+(x1**2)*theta[1]+x2*theta[2]
ax2.plot_surface(x1,x2,x3,rstride=1,cstride=1)
plt.show()
