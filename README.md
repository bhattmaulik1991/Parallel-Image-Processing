### Parallet Image Processing

In this project two different techniques: OpenMPI and PyCuda

# Denoising Images using Python OpenMPI

We tried denoising 20 images. The images were processed serially intially, and later with OpenMPI.

mpirun -n 2 python parallel-image-processing.py

https://www.nesi.org.nz/sites/default/files/mpi-in-python.pdf

<img src="https://github.com/bhattmaulik1991/cmpe275Proj2/blob/master/1.png" />

# Converting colored images to Black&White using PyCuda

We tried converting 100 images. The images were processed serially and using pycuda (Nvidia GEFORCE 940Mx).

python serialBW.py
python cudaBW.py

<img src="https://github.com/bhattmaulik1991/cmpe275Proj2/blob/master/2.png" />
