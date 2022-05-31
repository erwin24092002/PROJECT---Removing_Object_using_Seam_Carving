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
    
    @jit
    def get_minimum_seam(self, emap):
        """Get a minimum energy seam from emap
        Input: 
            np.array(h x w): a energy map
        Function return:
            np.array(h): a minimum energy seam of energy map
        """
        # Generate seam map
        smap = self.gen_smap(emap) 
        
        # Get seam
        seam = []
        h, w = smap.shape
        index = np.argmin(smap[h-1, :])
        seam.append(index)
        for i in range(h-2, -1, -1):
            if index == 0:
                index = index + np.argmin(smap[i, index:index+2])
            elif index == w-1:
                index = index - 1 +  np.argmin(smap[i, index-1:index+1])
            else: 
                index = index - 1 + np.argmin(smap[i, index-1:index+2])
            seam.append(index)
        return np.array(seam)[::-1]
    
    @jit
    def remove_seam(self, seam):
        """
        Input:
            arr(h) - a seam 
        Function return:
            arr(h x w x c) - an image with the deleted seam 
        """
        h, w, c = self.new_img.shape 
        new_img = np.zeros(shape=(h, w-1, c))
        for i in range(0, h):
            new_img[i, :seam[i], :] = self.new_img[i, :seam[i], :]
            new_img[i, seam[i]:, :] = self.new_img[i, seam[i]+1:, :]
        new_img = new_img.astype(np.uint8)
        return new_img
    
