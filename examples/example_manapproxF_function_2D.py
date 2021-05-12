import manapprox
import numpy as np
import matplotlib.pyplot as plt

s = 1
N = 10
data  = np.concatenate((np.linspace(0,1,N).reshape(1,N),np.zeros((1,N))),0)
f_data  = np.cos(np.linspace(0,1,N))
ma = manapprox.ManApproxF(data, f_data)
ma.manifold_dim = 1
ma.poly_deg = 2
ma.sparse_factor = 3
ma.calculateSigma()
ma.createTree()
print(ma)

point = np.array([0.5,0.3])
ma.calculateSigmaFromPoint(point)
print("New sigma:", ma.sigma)
projected_p, approximated_p = ma.approximateAtPoints(point)


print("projected point: ",projected_p)
print("mmm")
plt.plot(data[0,:], f_data, '.b')
plt.plot([point[0]], [point[1]], 'xr')
plt.plot([0.5], [approximated_p[0]], 'xg')
plt.show()
