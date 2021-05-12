import manapprox
import numpy as np
import matplotlib.pyplot as plt

#### Generate data
r = 5
s = 2.5
N = 5000

#rs = r + np.random.rand(N)*s*2 - s  # Not uniform distibution
theta = np.random.rand(N)*2*np.pi

A = 2/((r+s)**2 - (r-s)**2)
rs = np.sqrt(2*np.random.rand(N)/A + (r-s)**2)

data  = np.array([rs*np.sin(theta), rs*np.cos(theta)])

#### running algorithm

ma = manapprox.ManApprox(data)
ma.manifold_dim = 1
ma.poly_deg = 2
ma.sparse_factor = 10
#mmls.calculateSigma()
ma.sigma = 3.5
ma.createTree()
print(ma)
point = np.array([-1,-6])
delta_x = 1
max_delta_y = 3
projected_p = []
move_with_polynomail_approx = False

projected_p = ma.genGeodesDirection(point, 1, num_of_steps = 30, step_size_x = delta_x)

##############################
####    PLOT 
##############################


c = plt.Circle((0,0),5, color ='b', lw = 1, fill=False)
plt.axis('off')
plt.gca().set_aspect('equal', adjustable='box')
plt.plot(data[0,:], data[1,:], '.y', markersize=2, alpha=0.16)
plt.plot([point[0]], [point[1]], '.r')
plt.gca().add_patch(c)
plt.arrow(point[0], point[1], projected_p[0][0] - point[0], projected_p[0][1] - point[1],
     head_width=0.2, head_length=0.15, overhang=0.3, width=0.001, color="r",linestyle = (0,(1,3)),
     head_starts_at_zero = False, length_includes_head = True)
for i in range(1, len(projected_p)):
    plt.arrow(projected_p[i-1][0], projected_p[i-1][1], projected_p[i][0] - projected_p[i-1][0], projected_p[i][1] - projected_p[i-1][1],
     head_width=0.3, head_length=0.2, overhang=0.3, width=0.001, color="k",head_starts_at_zero = False, length_includes_head = True)
    #plt.plot([projected_p[i-1][0]], [projected_p[i-1][1]], '.g')
    #plt.title("iter: %d"%i)
    plt.draw()
    plt.pause(.05)

#plt.savefig("geoWalk_circle.svg", bbox_inches='tight')
plt.show()
print("")