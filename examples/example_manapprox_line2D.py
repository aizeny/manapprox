import manapprox
import numpy as np
import matplotlib.pyplot as plt

s = 1
N = 10
data  = np.concatenate((np.linspace(0,1,N).reshape(1,N),np.zeros((1,N))),0)
ma = manapprox.ManApprox(data)
ma.manifold_dim = 1
ma.poly_deg = 1
ma.calculateSigma()
ma.createTree()
print(ma)

point = np.array([0.5,0.3])
ma.calculateSigmaFromPoint(point)
print("New sigma:", ma.sigma)
projected_p = ma.projectPoints(point)


print("projected point: ",projected_p)
print("mmm")
plt.plot(data[0,:], data[1,:], '.b')
plt.plot([point[0]], [point[1]], '.r')
plt.plot([projected_p[0]], [projected_p[1]], 'xg')
plt.show()
