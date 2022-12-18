#!/usr/bin/env python
'''
===============================================================================
# grabcut

A simple program for interactively removing the background from an image using
the grab cut algorithm and OpenCV.

This code was derived from the Grab Cut example from the OpenCV project.

## Usage
    grabcut.py <input> [output]

## Operation

At startup, two windows will appear, one for input and one for output.

To start, in input window, draw a circle around the object using mouse right
button.  For finer touch-ups, press any of the keys below and draw circles on
the areas you want.  Finally, press 's' to save the result.

## Keys
  * 0 - Select areas of sure background
  * 1 - Select areas of sure foreground
  * 2 - Select areas of probable background
  * 3 - Select areas of probable foreground
  * n - Update the segmentation
  * r - Reset the setup
  * s - Save the result
  * q - Quit
===============================================================================
'''

# Python 2/3 compatibility
from __future__ import print_function
from copy import deepcopy
import numpy as np
import cv2 as cv
import sys
import os.path as osp
import os

def alpha_blend(image,mask, alpha = 0.5):     
    alpha_blend = image.copy()
    alpha_blend[:,:,2][mask!=0] =  alpha_blend[:,:,2][mask!=0] * alpha + mask[mask!=0] * (1-alpha)
    return alpha_blend

class App():
    BLUE  = [255, 0, 0]       # circle color
    RED   = [0, 0, 255]       # PR BG
    GREEN = [0, 255, 0]       # PR FG
    BLACK = [0, 0, 0]         # sure BG
    WHITE = [255, 255, 255]   # sure FG

    DRAW_BG    = {'color' : BLACK, 'val' : 0}
    DRAW_FG    = {'color' : WHITE, 'val' : 1}
    DRAW_PR_FG = {'color' : GREEN, 'val' : 3}
    DRAW_PR_BG = {'color' : RED,   'val' : 2}

    thickness  = 3


    def onmouse(self, event, x,y, flags, param):
        #Draw circle
        if event == cv.EVENT_RBUTTONDOWN:
            self.circle = True
            self.ix, self.iy = x,y

        elif event == cv.EVENT_MOUSEMOVE:
            if self.circle == True:
                radius = int(np.sqrt((self.ix -x)**2 + (self.iy - y)**2 ))
                self.input = self.copy.copy()
                cv.circle(self.input, (self.ix, self.iy), radius, self.BLUE,
                             2)
                self.circ = (min(self.ix, x), min(self.iy, y), radius )
                self.rect_or_mask = 0

        elif event == cv.EVENT_RBUTTONUP:
            self.circle = False
            self.rect_over = True
            radius = int(np.sqrt((self.ix -x)**2 + (self.iy - y)**2 ))
            cv.circle(self.input, (self.ix, self.iy), radius, self.BLUE, 2)
            sure_fg_radius = int(radius * 0.75)
            probable_fg_radius = int(radius * 1.1)
            probable_bg_radius = int(radius * 1.2)
            probable_bg = 2
            probable_fg = 3
            sure_fg = 1

            cv.circle(self.mask, (self.ix, self.iy), probable_bg_radius, probable_bg, -1)
            cv.circle(self.mask, (self.ix, self.iy), probable_fg_radius, probable_fg, -1)
            cv.circle(self.mask, (self.ix, self.iy), sure_fg_radius, sure_fg, -1)


            # cv.circle(self.mask, (self.ix, self.iy), radius, self.value['val'], -1)
            self.circ = (min(self.ix, x), min(self.iy, y), probable_bg_radius )
            self.rect_or_mask = 0
            self.segment()

        # Draw touchup curves
        if self.eraser:
            if event == cv.EVENT_LBUTTONDOWN:
                if not self.rect_over: print('First draw a circle')

                else:
                    
                    self.drawing = True
                    cv.circle(self.input, (x,y), self.eraser_width,
                            self.value['color'], -1)
                    cv.circle(self.mask, (x,y), self.eraser_width,
                            0, -1)

            elif event == cv.EVENT_MOUSEMOVE:
                if self.drawing == True:
                    cv.circle(self.input, (x, y), self.eraser_width,
                            self.value['color'], -1)
                            
                    cv.circle(self.mask, (x, y), self.eraser_width,
                            0, -1)

            elif event == cv.EVENT_LBUTTONUP:
                if self.drawing == True:
                    self.drawing = False
                    # cv.circle(self.input, (x, y), self.thickness,
                #           self.value['color'], -1)
                # cv.circle(self.mask, (x, y), self.thickness,
                #           self.value['val'], -1)
                # self.segment()tr


    def reset(self):
        print('Resetting')
        self.eraser = False
        self.eraser_width = 5
        self.circ = (0, 0, 1)
        self.drawing = False
        self.circle = False
        self.rect_or_mask = 100
        self.rect_over = False
        self.value = self.DRAW_FG

        self.input = self.copy.copy()
        self.mask = np.zeros(self.input.shape[:2], dtype = np.uint8)
        self.output = np.zeros(self.input.shape, np.uint8)


    def crop_to_alpha(self, img):
        x, y = self.alpha.nonzero()
        if len(x) == 0 or len(y) == 0: return img
        return img[np.min(x) : np.max(x), np.min(y) : np.max(y)]


    def save(self):
        new_mask = np.where((self.mask!=0)&(self.mask!=2), 255, 0)
        alpha_blend_im = alpha_blend(self.copy, new_mask)
        cv.imwrite(self.outfile, new_mask)
        cv.imwrite(self.outfile_alpha_blend, alpha_blend_im)
        print('Saved')
    

    def segment(self):
        try:
    

            bgdmodel = np.zeros((1, 65), np.float64)
            fgdmodel = np.zeros((1, 65), np.float64)
            xmin = max(0, self.circ[0] -   self.circ[2] - 50)
            ymin = max(0, self.circ[1] -   self.circ[2] - 50)
            xmax = min(self.copy.shape[1], self.circ[0] + self.circ[2] + 50)
            ymax = min(self.copy.shape[0], self.circ[1] + self.circ[2] + 50)
            patch = self.copy[ymin:ymax,xmin:xmax]
            patch_mask = self.mask[ymin:ymax,xmin:xmax]

            cv.grabCut(patch, patch_mask, None, bgdmodel, fgdmodel, 10,cv.GC_INIT_WITH_MASK)
            self.mask[ymin:ymax,xmin:xmax] = patch_mask
            new_mask = np.where((self.mask!=0)&(self.mask!=2), 255, 0)
            self.input = self.copy.copy()
            self.input  = alpha_blend(self.input , new_mask)

        except:
            import traceback
            traceback.print_exc()


    def load(self):
        self.outfile = 'grabcut.png'
        self.outfile_alpha_blend = 'grabcut.png'
        # if len(sys.argv) == 2: filename = sys.argv[1]
        # elif len(sys.argv) == 3: filename, self.outfile = sys.argv[1:3]
        # else: raise Exception('Usage: grabcut.py <input> [output]')
        src_dir = 'gt/source'
        dst_dir = 'gt/annotations'
        srcfiles = os.listdir(src_dir)
        annfiles = os.listdir(dst_dir)
        filename = None
        for file in srcfiles:
            if file not in annfiles:
                filename = osp.join(src_dir, file)
                self.outfile  = osp.join(dst_dir, file)
                fname = osp.splitext(file)[0] + '_blend.jpg'
                self.outfile_alpha_blend = osp.join(dst_dir, fname)
                break
        assert filename is not None
        self.input  = cv.imread(filename)
        self.copy   = self.input.copy()             # a copy of original image
        self.mask   = np.zeros(self.input.shape[:2], dtype = np.uint8)
        self.output = np.zeros(self.input.shape, np.uint8)
        self.alpha  = np.zeros(self.input.shape[:2], dtype = np.uint8)


    def run(self):
        self.load()
        self.reset()

        # Input and output windows
        cv.namedWindow('output', cv.WINDOW_GUI_NORMAL)
        cv.namedWindow('input', cv.WINDOW_GUI_NORMAL)
        cv.setMouseCallback('input', self.onmouse)
        cv.moveWindow('input', self.input.shape[1] + 10, 90)

        print('Draw a circle around the object using right mouse button')

        while True:
            cv.imshow('output', self.output)
            cv.imshow('input',  self.input)
            k = cv.waitKey(1)

            # Key bindings
            if k == 27 or k == ord('q'): break # exit
            elif k == ord('0'): self.value = self.DRAW_BG
            elif k == ord('1'): self.value = self.DRAW_FG
            elif k == ord('2'): self.value = self.DRAW_PR_BG
            elif k == ord('3'): self.value = self.DRAW_PR_FG
            elif k == ord('e'): self.eraser = not self.eraser
            elif k == ord('+'): self.eraser_width +=5            
            elif k == ord('-'): self.eraser_width = min(5, self.eraser_width -5)
            elif k == ord('s'): self.save()
            elif k == ord('r'): self.reset()
            elif k == ord('n'): self.segment()
            else: continue

            self.alpha = np.where((self.mask == 1) + (self.mask == 3), 255,
                                  0).astype('uint8')
            img = cv.bitwise_and(self.copy, self.copy, mask = self.alpha)
            self.output = self.crop_to_alpha(img)




if __name__ == '__main__':
    print(__doc__)
    App().run()
    cv.destroyAllWindows()
