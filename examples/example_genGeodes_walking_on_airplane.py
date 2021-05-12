import manapprox
import numpy as np
import numpy.random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib
import PIL.Image as IM
import glob
import time
import pickle
import imageio
import os
import string
import random
from stl import mesh # pip install numpy-stl
from mpl_toolkits import mplot3d
from matplotlib.colors import LightSource
import ntpath
import skimage
import skimage.transform


def genRotatedProjs1D(dir_name_clean, dir_name_final, stl_filename, N, noise_STD = 0, init3Drot = [1,0,0], add_rand_name = True, downsample = 1, slice_pixels = 0):
    """
    genRotatedProjs1D(output_dir_name, stl_filename, N, noise_STD = 0)

    Generates images from different 1-D angles of a 3d volume
    """
    # Check if clean images exsist. If no, generate them 
    if glob.glob(dir_name_clean+'*.png') == []:
        # Create a new plot
        figure = plt.figure()
        axes = mplot3d.Axes3D(figure)

        # Load the STL files and add the vectors to the plot
        your_mesh = mesh.Mesh.from_file(stl_filename)
        your_mesh.rotate(np.array([1,0,0]), -np.pi/2)
        axes.add_collection3d(mplot3d.art3d.Poly3DCollection(your_mesh.vectors))

        # Auto scale to the mesh size
        #scale = your_mesh.points.flatten(-1)
        scale = your_mesh.points.flatten()
        axes.auto_scale_xyz(scale, scale, scale)

        plt.axis('off')
        files_name = stl_filename.split('/')[-1].split('.')[0]
        # Show the plot to the screen and save png
        for angle in np.linspace(0,360,N):
                axes.view_init(30, angle)
                plt.draw()
                plt.savefig(dir_name_clean+files_name+"_ang{:03.2f}.png".format(angle))
                plt.pause(.001)
        plt.show()

    # Add noise
    changeImages(dir_name_clean, dir_name_final, downsample = downsample, add_rand_name = add_rand_name, added_noise_std = noise_STD, slice_pixels=slice_pixels)


def changeImages(input_dir_name, output_dir_name, downsample = 1, add_rand_name = False, added_noise_std = 0, slice_pixels = 0):
    files = glob.glob(input_dir_name+"*.png")
    for f_name in files:
        im = np.asarray(IM.open(f_name).convert(mode='L'), dtype=np.double)/255
        if slice_pixels !=0 :
            im = im[slice_pixels:-slice_pixels, slice_pixels:-slice_pixels]
        # Downsampling
        im_ds = skimage.transform.rescale(im,1./downsample)
        # Adding noise
        im_wnoise = im_ds + added_noise_std* np.random.randn(im_ds.shape[0],im_ds.shape[1])
        # Removing values larger then 1 and smaller then 0
        im_wnoise = np.min((np.ones_like(im_wnoise),im_wnoise),0)
        im_wnoise = np.max((np.zeros_like(im_wnoise),im_wnoise),0)
        #IM.fromarray((im*255).astype('int'), (im.shape[0],im.shape[1])).save(f_name)
        xx = IM.fromarray((im_wnoise*255).astype('int8'), mode='L')
        s_f_name = os.path.basename(f_name)
        if add_rand_name:
            randtit = ''.join(random.choices(string.ascii_uppercase + string.digits, k=5)) 
            s_f_name = s_f_name.split('_')[0] +'_'+ randtit +'_'+ s_f_name.split('_')[1]
            
        xx.save(output_dir_name + s_f_name)


def loadData(data_lib):
    im_files = glob.glob(data_lib+'/*.png')

    temp = np.asarray(IM.open(im_files[0]).convert(mode='L'), dtype=np.double)
    IM_SIZE_x = temp.shape[0]
    IM_SIZE_y = temp.shape[1]

    data = np.zeros((IM_SIZE_x*IM_SIZE_y,len(im_files)))
    for fname, i in zip(im_files, range(len(im_files))):
        data[:,i] = np.asarray(IM.open(fname).convert(mode='L'), dtype=np.double).flatten()/255
    print("Data loaded...")
    return data, IM_SIZE_x,IM_SIZE_y


dir_name_clean = './airplaneData_N2000/'
dir_name_final = './airplaneData_N2000_Sig0_ds2/'
STL_FILENAME = './airplane.stl'
data_lib = './airplaneData_N2000_Sig0'


############################################
###     Generate data from STL
############################################

genRotatedProjs1D(dir_name_clean, dir_name_final, STL_FILENAME, 2000, noise_STD = 0, init3Drot = [1,0,0], add_rand_name=False, downsample=2, slice_pixels = 30)

############################################
###     Loading data
############################################
data, IM_SIZE_x,IM_SIZE_y = loadData(data_lib)


# Initializing MMLS class
ma = manapprox.ManApprox(data)
ma.manifold_dim = 1
ma.poly_deg = 3
t =time.time()
ma.createTree()
print("Time for building tree: %f"%(time.time() - t))
t =time.time()
ma.calculateSigma() 
print("Time for calculateSigma: %f"%(time.time() - t))
ma.sparse_factor = 5
ma.recompute_sigma = True
ma.initSVD = True
ma.thresholdH = ma.sigma*0.05
print(ma)

# Initial point for movement
point = data[:,20]
delta_x = 2.5 # The shift between two steps on the tangent
delta_y = 5 # the maximal shift between the current projection and the new initial point
num_of_steps = 700
projected_p = ma.genGeodesDirection(point, 1, num_of_steps = num_of_steps, step_size_x = delta_x, step_size_y = delta_y, verbose = True)

print("")
for i,p in enumerate(projected_p):
    plt.imshow(1-p.reshape([IM_SIZE_x,IM_SIZE_y]), cmap='Blues')
    plt.title("iter: %d"%i)
    plt.draw()
    plt.pause(.001)
    plt.cla()

plt.show()



