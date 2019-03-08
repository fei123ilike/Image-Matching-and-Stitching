from q2 import briefLite, briefMatch
from q3 import computeH, computeHnorm, computeHransac, compositeH, HarryPotterize

import skimage.color
import skimage.io
import numpy as np
import math

# make a test case!
# you should write your own
# Create H, x1 and x2 whose answer you know
# Make sure you can recover it!
H, x1, x2 = None, None, None
img1 = skimage.io.imread('../data/pnc1.png')
im1 = skimage.color.rgb2gray(img1)
img2 = skimage.io.imread('../data/pnc0.png')
im2 = skimage.color.rgb2gray(img2)
locs1, desc1 = briefLite(im1)
locs2, desc2 = briefLite(im2)
matches = briefMatch(desc1,desc2,ratio=0.8)

l1 = [locs1[matches[:,0], 0:2]]
l2 = [locs2[matches[:,1], 0:2]]

x1 = np.array(l1)
x2 = np.array(l2)
x1 = x1.squeeze()
x2 = x2.squeeze()
# 3.1
x1 = np.array([[25,100],[100,25],[50,150],[300,300]])
x2 = np.array([[45,150],[120,75],[70,200],[320,350]])
H2to1 = computeH(x1, x1)
H2to1 = H2to1/H2to1[2,2]
print('should be identity\n',H2to1,'\n')



H2to1 = computeH(x2, x1)
H2to1 = H2to1/H2to1[2,2]
print('normal\n',H2to1,'\n')

# 3.2
H2to1 = computeHnorm(x2, x1)
H2to1 = H2to1/H2to1[2,2]
print('normalized\n',H2to1,'\n')

# 3.3 
bestH2to1, inliers = computeHransac(x1, x2)
bestH2to1 = bestH2to1/bestH2to1[2,2]
print('ransac\n',bestH2to1,'\n')

# 3.4
HarryPotterize()