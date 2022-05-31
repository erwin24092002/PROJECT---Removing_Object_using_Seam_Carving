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
        
    
