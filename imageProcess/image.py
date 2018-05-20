import numpy as np
from skimage import data, img_as_float 
import skimage.filters
from skimage import data, img_as_float
from skimage import io
import os.path
import time
from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral,
                                 denoise_wavelet, estimate_sigma)
curPath = os.path.abspath(os.path.curdir) 
noisyDir = os.path.join(curPath,'noisy') 
denoisedDir = os.path.join(curPath,'denoised')
def loop(imgFiles):
    for f in imgFiles:
        img = img_as_float(data.load(os.path.join(noisyDir,f)))
        startTime = time.time()
        dim = img.shape
        img = denoise_bilateral(img, sigma_color=0.05, sigma_spatial=15,  multichannel=True)
        io.imsave(os.path.join(denoisedDir,f), img)
        #skimage.io.imsave(os.path.join(denoisedDir,f), img)
        print("Took %f seconds for %s" %(time.time() - startTime, f))

def serial():
    total_start_time = time.time()
    imgFiles = ["%.4d.jpg"%x for x in range(1,21)]
    loop(imgFiles)
    print("Total time %f seconds" %(time.time() - total_start_time))

if __name__=='__main__': 
    serial()