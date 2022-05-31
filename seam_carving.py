import cv2
import imageio 
import imutils 
import matplotlib.pyplot as plt 
import numpy as np 
from scipy import ndimage as ndi 
from tqdm import tqdm 
from numba import jit 

REMOVAL_SEAM_COLOR = np.array([0, 0, 255])  # Color of seam when visualizing
INSERTED_SEAM_COLOR = np.array([0, 255, 0])  # Color of seam when visualizing

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
        """ Remove input seam from image
        Input:
            np.array(h) - a seam  
        Function return:
            np.array(h x w-1 x c) - an image after removing input seam 
        """
        h, w, c = self.new_img.shape 
        new_img = np.zeros(shape=(h, w-1, c))
        for i in range(0, h):
            new_img[i, :seam[i], :] = self.new_img[i, :seam[i], :]
            new_img[i, seam[i]:, :] = self.new_img[i, seam[i]+1:, :]
        new_img = new_img.astype(np.uint8)
        return new_img
    
    @jit
    def insert_seam(self, seam):
        """ Insert new seam to image
        Input:
            np.array(h) - a seam  
        Function return:
            np.array(h x w+1 x c) - an image after inserting new seam 
        """
        h, w, c = self.new_img.shape 
        new_img = np.zeros(shape=(h, w+1, c))
        for i in range(0, h):
            new_img[i, :seam[i], :] = self.new_img[i, :seam[i], :]
            new_img[i, seam[i]+1:, :] = self.new_img[i, seam[i]:, :]
            if seam[i] == 0:
                new_img[i, seam[i], :] = self.new_img[i, seam[i]+1, :]
            elif seam[i] == w-1:
                new_img[i, seam[i], :] = self.new_img[i, seam[i]-1, :]
            else:
                new_img[i, seam[i], :] = (self.new_img[i, seam[i]-1, :].astype(np.int32)
                                          + self.new_img[i, seam[i]+1, :].astype(np.int32)) / 2
        new_img = new_img.astype(np.uint8)
        return new_img

    @jit
    def resize(self, new_size=(0, 0)): 
        self.new_img[:] = self.img[:]
        h, w, c = self.new_img.shape
        new_h, new_w = new_size
        delta_h = new_h - h
        delta_w = new_w - w 
        
        if delta_w > 0:
            for i in tqdm(range(delta_w), desc="Horizontal Processing"):
                emap = self.gen_emap()
                seam = self.get_minimum_seam(emap)
                self.new_img = self.insert_seam(seam)
                self.sliders.append(self.visual_seam(seam, color=INSERTED_SEAM_COLOR))
        elif delta_w < 0: 
            delta_w = abs(delta_w)
            for i in tqdm(range(delta_w), desc="Horizontal Processing"):
                emap = self.gen_emap()
                seam = self.get_minimum_seam(emap)
                self.sliders.append(self.visual_seam(seam))
                self.new_img = self.remove_seam(seam)
        
        if delta_h > 0: 
            delta_h = abs(delta_h)
            self.new_img = imutils.rotate_bound(self.new_img, angle=90)
            for i in tqdm(range(delta_h), desc="Vertical Processing"):
                emap = self.gen_emap()
                seam = self.get_minimum_seam(emap)
                self.new_img = self.insert_seam(seam)
                self.sliders.append(self.visual_seam(seam, color=INSERTED_SEAM_COLOR))
                self.sliders[len(self.sliders)-1] = imutils.rotate_bound(self.sliders[len(self.sliders)-1], angle=-90)
            self.new_img = imutils.rotate_bound(self.new_img, angle=-90)
        elif delta_h < 0:
            delta_h = abs(delta_h)
            self.new_img = imutils.rotate_bound(self.new_img, angle=90)
            for i in tqdm(range(delta_h), desc="Vertical Processing"):
                emap = self.gen_emap()
                seam = self.get_minimum_seam(emap)
                self.sliders.append(self.visual_seam(seam))
                self.sliders[len(self.sliders)-1] = imutils.rotate_bound(self.sliders[len(self.sliders)-1], angle=-90)
                self.new_img = self.remove_seam(seam)
            self.new_img = imutils.rotate_bound(self.new_img, angle=-90)
        
        new_img = np.zeros_like(self.new_img).astype(np.uint8)
        new_img[:] = self.new_img[:]
        return new_img
    
    @jit
    def visual_seam(self, seam, color=REMOVAL_SEAM_COLOR):
        """
        Input:
            arr(h) - a seam 
        Function return:
            arr(h x w x c) - an image with the seam line colored
        """
        h, w, c = self.new_img.shape
        new_img = np.zeros_like(self.new_img)
        new_img[:] = self.new_img[:]
        for i in range(0, h):
            new_img[i, seam[i], :] = color
        new_img = new_img.astype(np.uint8)
        return new_img
    
    def visual_process(self, save_path=''):
        """
        Input:
            link to save process.gif file
        Function collects processing states stored in self.sliders to form a .gif file and save it at save_path
        """
        print("Waiting for Visualizing Process...")
        h, w, c = self.img.shape 
        new_h, new_w, new_c = self.new_img.shape
        frames = [np.zeros(shape=(max(h, new_h), max(w, new_w), c)).astype(np.uint8)]
        for slider in self.sliders: 
            slider = cv2.cvtColor(slider, cv2.COLOR_BGR2RGB)
            frames.append(imageio.core.util.Array(slider))
        imageio.mimsave(save_path, frames)
        print("Completed process.gif creating at {0}".format(save_path))
    
    def visual_result(self, save_path=''):
        """
        Input: 
            link to save result of process 
        Function save Result at save_path
        Result includes 4 image
            1. source image 
            2. energy map of source image
            3. seam map of energy map above
            4. the image after process - new image
        """
        img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        result = cv2.cvtColor(self.new_img, cv2.COLOR_BGR2RGB)
        plt.subplot(1, 2, 1), plt.imshow(img, cmap='gray'), plt.title("Image ({0};{1})".format(img.shape[0], img.shape[1]))
        plt.subplot(1, 2, 2), plt.imshow(result, cmap='gray'), plt.title("Resized Image ({0};{1})".format(result.shape[0], result.shape[1]))
        if save_path != '':
            plt.savefig(save_path)
        plt.show()