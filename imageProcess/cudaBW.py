import numpy as np
from skimage import data, img_as_float 
import skimage.filters
from skimage import data, img_as_float
from skimage import io
import os.path
import time
from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral,
                                 denoise_wavelet, estimate_sigma)
import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule

curPath = os.path.abspath(os.path.curdir) 
noisyDir = os.path.join(curPath,'colored') 
denoisedDir = os.path.join(curPath,'bw')

mod = SourceModule("""
#include <stdio.h>
__global__ void processimage(float *dest, float *img)
{
    int x = threadIdx.x+ blockIdx.x* blockDim.x;
    int y = threadIdx.y+ blockIdx.y* blockDim.y;
    float val;
    val = (img[x + y + 0] + img[x + y + 1] + img[x + y + 2])/3;
    dest[x + y + 0] = val;
    dest[x + y + 1] = val;
    dest[x + y + 2] = val;
}
""")
processimage = mod.get_function("processimage")

def parallelCuda():
    total_start_time = time.time()
    imgFloats = []
    startTime = time.time()
    for y in range(0,20):
        for x in range(1,6):
            imgFloats = np.array(img_as_float(data.load(os.path.join(noisyDir,"%.4d.jpg"%x))), dtype=np.float32)
            dest = imgFloats
            processimage(drv.Out(dest), drv.In(imgFloats), block=(1024,1,1), grid=(128,1))
            io.imsave(os.path.join(denoisedDir,"%.4d.jpg"%(x*y)), dest)
    print("Total time %f seconds" %(time.time() - total_start_time))

if __name__=='__main__': 
    parallelCuda()