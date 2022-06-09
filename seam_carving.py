import cv2
import imageio 
import imutils 
import pygame
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
        self.isgray = len(img.shape) == 2
        self.sliders = []  
        self.u_kernel = np.array([[0., 0., 0.], [-1., 0., 1.], [0., 0., 0.]], dtype=np.float64)
        self.l_kernel = np.array([[0., 1., 0.], [-1., 0., 0.], [0., 0., 0.]], dtype=np.float64)
        self.r_kernel = np.array([[0., 1., 0.], [0., 0., 1.], [0., 0., 0.]], dtype=np.float64)
    
    @jit
    def gen_emap(self):
        """Generate an energy map using Gradient magnitude
        Args:
        Returns:
            np.array: an energy map (emap) of input image 
        """
        Gx = ndi.convolve1d(self.new_img, np.array([1, 0, -1]), axis=1, mode='wrap')
        Gy = ndi.convolve1d(self.new_img, np.array([1, 0, -1]), axis=0, mode='wrap')
        if self.isgray:
            return np.sqrt(Gx**2 + Gy**2)
        else:
            return np.sqrt(np.sum(Gx**2, axis=2) + np.sum(Gy**2, axis=2))

    @jit
    def calc_backward_emap(self, emap):
        """Generate an seam map from energy map
        Args:
            np.array(h, w): an energy map
        Returns:
            np.array(h, w): an seam map (bemap) of input energy map 
        """ 
        h, w = emap.shape 
        bemap = emap.copy()
        for row in range(1, h):
            for col in range(0, w):
                bemap[row, col] = min(bemap[row-1, max(0, col-1):min(col+2, w)]) + bemap[row, col]
        return bemap

    @jit
    def calc_forward_emap(self, emap):
        under_forward_diff = self.calc_forward_diff(self.u_kernel)
        left_forward_diff = self.calc_forward_diff(self.l_kernel)
        right_forward_diff = self.calc_forward_diff(self.r_kernel)
        h, w = emap.shape
        femap = emap.copy()
        for row in range(1, h):
            for col in range(w):
                if col == 0:
                    r_cost = femap[row - 1, col + 1] + under_forward_diff[row - 1, col + 1] + right_forward_diff[row - 1, col + 1]
                    u_cost = femap[row - 1, col] + under_forward_diff[row - 1, col]
                    femap[row, col] = femap[row, col] + min(r_cost, u_cost)
                elif col == w - 1:
                    l_cost = femap[row - 1, col - 1] + under_forward_diff[row - 1, col - 1] + left_forward_diff[row - 1, col - 1]
                    u_cost = femap[row - 1, col] + under_forward_diff[row - 1, col]
                    femap[row, col] = femap[row, col] + min(l_cost, u_cost)
                else:
                    l_cost = femap[row - 1, col - 1] + under_forward_diff[row - 1, col - 1] + left_forward_diff[row - 1, col - 1]
                    r_cost = femap[row - 1, col + 1] + under_forward_diff[row - 1, col + 1] + right_forward_diff[row - 1, col + 1]
                    u_cost = femap[row - 1, col] + under_forward_diff[row - 1, col]
                    femap[row, col] = femap[row, col] + min(l_cost, r_cost, u_cost)
        return femap
    
    @jit
    def calc_forward_diff(self, kernel):
        forward_diff = 0
        if self.isgray:
            forward_diff = np.absolute(cv2.filter2D(self.new_img, -1, kernel)) 
        else:
            b, g, r = cv2.split(self.new_img)
            forward_diff = (np.absolute(cv2.filter2D(b, -1, kernel)) 
                            + np.absolute(cv2.filter2D(g, -1, kernel)) 
                            + np.absolute(cv2.filter2D(r, -1, kernel)))
        return forward_diff

    @jit
    def get_minimum_seam(self, emap):
        """Get a minimum energy seam from emap
        Input: 
            np.array(h x w): a energy map
        Function return:
            np.array(h): a minimum energy seam of energy map
        """
        # Generate seam map
        smap = self.calc_forward_emap(emap) 
        
        # Get seam
        seam = []
        h, w = smap.shape
        index = np.argmin(smap[h-1, :])
        for row in range(h-1, -1, -1):
            if index == 0:
                index = np.argmin(smap[row, :2])
            else: 
                index = index - 1 + np.argmin(smap[row, max(0,index-1):min(index+2, w)])
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
        h, w = self.new_img.shape[0:2]
        new_img = 0
        if self.isgray:
            new_img = np.zeros(shape=(h, w-1))
        else: 
            new_img = np.zeros(shape=(h, w-1, 3))
        for row in range(0, h):
            col = seam[row]
            new_img[row, :col] = self.new_img[row, :col]
            new_img[row, col:] = self.new_img[row, col+1:]
        return new_img.astype(np.uint8)
    
    @jit
    def remove_seam_on_mask(self, seam, mask):
        h, w = mask.shape 
        new_mask = np.zeros(shape=(h, w-1))
        for row in range(0, h):
            col = seam[row]
            new_mask[row, :col] = mask[row, :col]
            new_mask[row, col:] = mask[row, col+1:]
        return new_mask
    
    @jit
    def insert_seam(self, seam):
        """
        Input:
            arr(h) - a seam 
        Function return:
            arr(h x w x c) - an image with the inserted seam 
        """
        h, w = self.new_img.shape[0:2] 
        new_img = 0
        if self.isgray:
            new_img = np.zeros(shape=(h, w+1))
        else: 
            new_img = np.zeros(shape=(h, w+1, 3))
        for row in range(0, h):
            col = seam[row]
            new_img[row, :col] = self.new_img[row, :col]
            new_img[row, col+1:] = self.new_img[row, col:]
            if col == 0:
                new_img[row, col] = new_img[row, col+1]
            else:
                new_img[row, col] = (new_img[row, col-1].astype(np.int32)
                                        + new_img[row, col+1].astype(np.int32)) / 2
        return new_img.astype(np.uint8)
        
    @jit
    def remove_object(self, removal_mask, protective_mask=None):
        self.removal_mask = removal_mask.copy()
        if protective_mask is not None:
            self.protective_mask = protective_mask.copy()
        
        rotate_flag = False
        dmask_w = len(set(np.where(self.removal_mask.T==255)[0]))
        dmask_h = len(set(np.where(self.removal_mask==255)[0]))
        dmask = dmask_w
        if dmask_h < dmask_w:
            dmask = dmask_h
            rotate_flag = True
        
        # Removing Object
        if rotate_flag: 
            self.new_img = imutils.rotate_bound(self.new_img, angle=90)
            self.removal_mask = imutils.rotate_bound(self.removal_mask, angle=90)
            if protective_mask is not None:
                self.protective_mask = imutils.rotate_bound(self.protective_mask, angle=90)
            
        for step in tqdm(range(dmask), desc='Removing Object'):
            emap = self.gen_emap()
            if protective_mask is not None:
                emap = np.where((self.protective_mask==255), 1000, emap)
            emap = np.where((self.removal_mask==255), -1000, emap)
            seam = self.get_minimum_seam(emap)
            self.sliders.append(self.visual_seam(seam, color=REMOVAL_SEAM_COLOR))
            if rotate_flag:
                self.sliders[len(self.sliders)-1] = imutils.rotate_bound(self.sliders[len(self.sliders)-1], angle=-90)
            self.new_img = self.remove_seam(seam)
            self.removal_mask = self.remove_seam_on_mask(seam, self.removal_mask)
            if protective_mask is not None:
                self.protective_mask = self.remove_seam_on_mask(seam, self.protective_mask)
        
        # Regaining orginal size    
        temp_img = self.new_img.copy()
        seam_record = []
        for step in tqdm(range(dmask), desc='Colecting Seams for Insertion'):
            emap = self.gen_emap()
            if protective_mask is not None:
                emap = np.where((self.protective_mask==255), 1000, emap)
            seam = self.get_minimum_seam(emap)
            seam_record.append(seam)
            self.new_img = self.remove_seam(seam)
            if protective_mask is not None:
                self.protective_mask = self.remove_seam_on_mask(seam, self.protective_mask)
            
        self.new_img = temp_img.copy()
        for step in tqdm(range(dmask), desc='Regaining Original Size'):
            seam = seam_record.pop(0)
            self.new_img = self.insert_seam(seam)
            self.sliders.append(self.visual_seam(seam, color=INSERTED_SEAM_COLOR))
            if rotate_flag:
                self.sliders[len(self.sliders)-1] = imutils.rotate_bound(self.sliders[len(self.sliders)-1], angle=-90)
            seam_record = self.update_seam_record(seam_record, seam)
        
        if rotate_flag: 
            self.new_img = imutils.rotate_bound(self.new_img, angle=-90)
        
        return self.new_img.copy()
    
    @jit
    def update_seam_record(self, seam_record, seamm):
        new_record = []
        for seam in seam_record:
            seam[np.where(seam>=seamm)] += 2 
            new_record.append(seam)
        return new_record
    
    @jit
    def visual_seam(self, seam, color=REMOVAL_SEAM_COLOR):
        """
        Input:
            arr(h) - a seam 
        Function return:
            arr(h x w x c) - an image with the seam line colored
        """
        if self.isgray:
            color = 0
        new_img = self.new_img.copy()
        h = self.new_img.shape[0]
        for row in range(0, h):
            new_img[row, seam[row]] = color
        return new_img.astype(np.uint8)
    
    def visual_process(self, save_path=''):
        """
        Input:
            link to save process.gif file
        Function collects processing states stored in self.sliders to form a .gif file and save it at save_path
        """
        print("Waiting for Visualizing Process...")
        h, w = self.img.shape[0:2] 
        new_h, new_w = self.new_img.shape[0:2]
        frames = []
        if self.isgray:
            frames.append(np.zeros(shape=(max(h, new_h), max(w, new_w))).astype(np.uint8))
            for slider in self.sliders: 
                frames.append(imageio.core.util.Array(slider))
        else: 
            frames = [np.zeros(shape=(max(h, new_h), max(w, new_w), 3)).astype(np.uint8)]
            for slider in self.sliders: 
                slider = cv2.cvtColor(slider, cv2.COLOR_BGR2RGB)
                frames.append(imageio.core.util.Array(slider))
        imageio.mimsave(save_path, frames)
        print("Completed process.gif creating at {0}".format(save_path))
        
    def get_mask(self, desc='Seam Carving'):
        """
        This function display image for user to choose areas which they want to delete.
        Args:
            img: Input image. Defaults to None.
        Returns:
            mask: 2D Binary mask. 
        """
        pygame.init()
        
        h, w = self.img.shape[0:2]
        temp_img = 0
        if self.isgray:
            temp_img = cv2.cvtColor(cv2.resize(self.img, (w//2, h//2)), cv2.COLOR_GRAY2RGB)
        else:
            temp_img = cv2.cvtColor(cv2.resize(self.img, (w//2, h//2)), cv2.COLOR_BGR2RGB)
        rotated_img = np.rot90(temp_img)

        pygame.display.set_caption(desc)
        window = pygame.display.set_mode((rotated_img.shape[0], rotated_img.shape[1]))
        background_surf = pygame.surfarray.make_surface(rotated_img).convert()
        background_surf = pygame.transform.flip(background_surf, True, False)

        fps = 120   
        clock = pygame.time.Clock()
        start = True
        flag = False
        pts = []

        mask = np.zeros((temp_img.shape[0], temp_img.shape[1]))
        while start:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    start = False
                    pygame.quit()
                    return cv2.resize(mask, (w, h))
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    flag = True
                    x, y = pygame.mouse.get_pos()
                    pygame.draw.rect(background_surf, (255, 0, 0), pygame.Rect(x, y, 1, 1))
                    pts.append(pygame.mouse.get_pos())
                elif flag and event.type == pygame.MOUSEMOTION:
                    x, y = pygame.mouse.get_pos()
                    pygame.draw.rect(background_surf, (255, 0, 0), pygame.Rect(x, y, 1, 1))
                    pts.append(pygame.mouse.get_pos())
                elif event.type == pygame.MOUSEBUTTONUP:
                    npar = np.array(pts)
                    pygame.draw.polygon(background_surf, (255, 255, 255), pts)
                    cv2.fillConvexPoly(mask, npar, 255)
                    cv2.imshow('mask', mask)
                    cv2.waitKey(0)
                    pts = []
                    flag = False
                
            window.blit(background_surf, (0, 0))
            pygame.display.update()
            clock.tick(fps)