import manapprox
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from time import time as _t

# CONSTS
########
N = 800
N_ratio = 0.1
fN_ratio = 0.5
DEBUG=False
color_scheme = "jet"
SPARSE_FACTOR = 20
DEG = 2
DIM = 1
np.random.seed(0)
# Data Preperation - Helix1 (Uniformly Sampled)
#----------------------------------------------
DIM = 1 # this is the intrinsic dimension of the data
num_pi = 4
h = num_pi*np.pi/N
theta = np.arange(-num_pi/2*np.pi,num_pi/2*np.pi,h)
r = 1
z = 1*theta
x = r * np.sin(theta)
y = r * np.cos(theta)


# we define N_ratio% noise at each coordinate
Mx = abs(max(x) - min(x))
My = abs(max(y) - min(y))
Mz = abs(max(z) - min(z))
total_M = np.sqrt(Mx**2+My**2+z**2)
noise_x = (0.5*total_M - np.random.rand(x.size)*total_M)*N_ratio
noise_y = (0.5*total_M - np.random.rand(y.size)*total_M)*N_ratio
noise_z = (0.5*total_M - np.random.rand(z.size)*total_M)*N_ratio
xN = x + noise_x.reshape(x.shape)
yN = y + noise_y.reshape(y.shape)
zN = z + noise_z.reshape(z.shape)
x_vec = np.asarray([xN,yN,zN])
x_vec_clean = np.asarray([x,y,z])


#############################
# TAKE 1 WITHOUT DOMAIN NOISE
#############################
# Define a function over the data (currently we take the z-axis)
f_data = z + (0.5 - np.random.rand(*z.shape))*fN_ratio*Mz
# Create the MMLS object
#-------------------------
ma_f = manapprox.ManApproxF(x_vec_clean, f_data)
ma_f.manifold_dim = DIM
ma_f.poly_deg = DEG
ma_f.sparse_factor = SPARSE_FACTOR
ma_f.calculateSigma()
ma_f.createTree()
print(ma_f)

# Approximate
#-------------
print("Approximating All Data points -- Take 1")
t0 = _t()
approximated_data = ma_f.approximateMany(ma_f.data, True)
print("Finshed projection after", _t()-t0, "secs")
projected_data = np.asarray(ma_f.approximated_points_res).T[0]

# Plot the data
#-------------------------
fig = plt.figure(figsize=plt.figaspect(0.5))
ax = fig.add_subplot(1,3,1,projection='3d')
eps = 0
axim = ax.scatter(x,y,z,c=z, cmap=color_scheme, s=100)
plt.title("Clean Sample")
fig.colorbar(axim)

ax = fig.add_subplot(1,3,2,projection='3d')
eps = 0
axim = ax.scatter(x_vec_clean[0],x_vec_clean[1],x_vec_clean[2],c=f_data, cmap=color_scheme, s=100)
plt.title("Noisy sample")
fig.colorbar(axim)

ax = fig.add_subplot(1,3,3,projection='3d')
eps = 0
axim = ax.scatter(projected_data[0],projected_data[1],projected_data[2],c=approximated_data.squeeze(), cmap=color_scheme, s=100)
plt.title("Approximated values")
fig.colorbar(axim)

#############################
# TAKE 2 WITH DOMAIN NOISE
#############################
# Create the MMLS object
#-------------------------
ma_f = manapprox.ManApproxF(x_vec, f_data)
ma_f.manifold_dim = DIM
ma_f.poly_deg = DEG
ma_f.sparse_factor = SPARSE_FACTOR
ma_f.calculateSigma()
ma_f.createTree()
print(ma_f)

# Approximate
#-------------
print("Approximating All Data points -- Take 2")
t0 = _t()
approximated_data = ma_f.approximateMany(ma_f.data, True)
print("Finshed projection after", _t()-t0, "secs")
projected_data = np.asarray(ma_f.approximated_points_res).T[0]

# Plot the data
#-------------------------
fig = plt.figure(figsize=plt.figaspect(0.5))
ax = fig.add_subplot(1,3,1,projection='3d')
eps = 0
axim = ax.scatter(x,y,z,c=z, cmap=color_scheme, s=100)
plt.title("Clean Sample")
fig.colorbar(axim)

ax = fig.add_subplot(1,3,2,projection='3d')
eps = 0
axim = ax.scatter(xN,yN,zN,c=f_data, cmap=color_scheme, s=100)
plt.title("Noisy sample")
fig.colorbar(axim)

ax = fig.add_subplot(1,3,3,projection='3d')
eps = 0
axim = ax.scatter(projected_data[0],projected_data[1],projected_data[2],c=approximated_data.squeeze(), cmap=color_scheme, s=100)
plt.title("Approximated values")
fig.colorbar(axim)

###########################################
# TAKE 3 WITHOUT DOMAIN NOISE - new samples
###########################################
# Create new noisy points
#------------------------
np.random.seed(10)
noise_x = (0.5*total_M - np.random.rand(x.size)*total_M)*N_ratio
noise_y = (0.5*total_M - np.random.rand(y.size)*total_M)*N_ratio
noise_z = (0.5*total_M - np.random.rand(z.size)*total_M)*N_ratio
new_xN = x + noise_x.reshape(x.shape)
new_yN = y + noise_y.reshape(y.shape)
new_zN = z + noise_z.reshape(z.shape)
new_x_vec = np.asarray([new_xN,new_yN,new_zN])

# Create the MMLS object
#-------------------------
ma_f = manapprox.ManApproxF(x_vec_clean, f_data)
ma_f.manifold_dim = DIM
ma_f.poly_deg = DEG
ma_f.sparse_factor = SPARSE_FACTOR
ma_f.calculateSigma()
ma_f.createTree()
print(ma_f)

# Approximate
#-------------
print("Approximating All Data points -- Take 3")
t0 = _t()
approximated_data = ma_f.approximateMany(new_x_vec, True)
print("Finshed projection after", _t()-t0, "secs")
projected_data = np.asarray(ma_f.approximated_points_res).T[0]

# Plot the data
#-------------------------
fig = plt.figure(figsize=plt.figaspect(0.5))
ax = fig.add_subplot(1,3,1,projection='3d')
eps = 0
axim = ax.scatter(x,y,z,c=f_data, cmap=color_scheme, s=100)
plt.title("Sample")
fig.colorbar(axim)

ax = fig.add_subplot(1,3,2,projection='3d')
eps = 0
ax.scatter(new_xN,new_yN,new_zN, s=100)
plt.title("New input")

ax = fig.add_subplot(1,3,3,projection='3d')
eps = 0
axim = ax.scatter(new_xN,new_yN,new_zN,c=approximated_data.squeeze(), cmap=color_scheme, s=100)
plt.title("Approximated values")
fig.colorbar(axim)

plt.show()

