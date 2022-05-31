import cv2
import imageio 
import imutils 
import matplotlib.pyplot as plt 
import numpy as np 
from scipy import ndimage as ndi 
from tqdm import tqdm 
from numba import jit 

class SeamCarving: 
    def __init__(self, img):
        self.img = img 
        self.new_img = img.copy() 
        self.sliders = []  
    
    @jit
    def gen_emap(self):
        """Generate an energy map using Gradient magnitude
        Args:
        Returns:
            np.array: an energy map (emap) of input image 
        """
        Gx = ndi.convolve1d(self.new_img, np.array([1, 0, -1]), axis=1, mode='wrap')
        Gy = ndi.convolve1d(self.new_img, np.array([1, 0, -1]), axis=0, mode='wrap')
        emap = np.sqrt(np.sum(Gx**2, axis=2) + np.sum(Gy**2, axis=2))
        return emap

    @jit
    def gen_smap(self, emap):
        """Generate an seam map from energy map
        Args:
            np.array(h, w): an energy map
        Returns:
            np.array(h, w): an seam map (smap) of input energy map 
        """ 
        h, w = emap.shape 
        smap = emap.copy()
        for i in range(1, h):
            for j in range(0, w):
                if j == 0:
                    smap[i, j] = min(smap[i-1, j:j+2]) + smap[i, j]
                elif j == w-1:
                    smap[i, j] = min(smap[i-1, j-1:j+1]) + smap[i, j]
                else: 
                    smap[i, j] = min(smap[i-1, j-1:j+2]) + smap[i, j]
        return smap
    
    
