import manapprox
import numpy as np
import matplotlib.pyplot as plt

r = 10
s = 1
N = 1000
rs = r + np.random.rand(N)*s*2 - s
theta = np.random.rand(N)*2*np.pi
data  = np.array([rs*np.sin(theta), rs*np.cos(theta)])
ma = manapprox.ManApprox(data)
ma.manifold_dim = 1
ma.poly_deg = 2
ma.calculateSigma()
ma.createTree()
print(ma)
ma.sigma = 3
point = np.array([11,1])
projected_p = ma.projectPoints(point)

plt.plot(data[0,:], data[1,:], '.b')
plt.plot([point[0]], [point[1]], 'or')
plt.plot([projected_p[0]], [projected_p[1]], 'og')
plt.show()
