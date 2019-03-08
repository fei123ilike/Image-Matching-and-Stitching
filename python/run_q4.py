from q4 import *
from q3 import *
from q2 import *
import numpy as np
import matplotlib.pyplot as plt
import skimage.color
import skimage.io
import PIL.Image
# Q 4.1
# load images into img1
# and img2
# compute H2to1
# please use any feature method you like that gives
# good results
img1 = skimage.io.imread('../data/pnc1.png')
img2 = skimage.io.imread('../data/pnc0.png')
#img1 = skimage.io.imread('../data/incline_L.png')
#img2 = skimage.io.imread('../data/incline_R.png')

bestH2to1 = None
# YOUR CODE HERE
im1 = skimage.color.rgb2gray(img1)
im2 = skimage.color.rgb2gray(img2)
   #get descriptors for img1 img2
descriptor_extractor=ORB(n_keypoints=600)
descriptor_extractor.detect_and_extract(im1)
key1=descriptor_extractor.keypoints
descriptor1 = descriptor_extractor.descriptors
    
descriptor_extractor=ORB(n_keypoints=600)
descriptor_extractor.detect_and_extract(im2)
key2=descriptor_extractor.keypoints
descriptor2 = descriptor_extractor.descriptors
#find match points for img1 and img2
match = match_descriptors(descriptor1, descriptor2,cross_check=True)
num=np.size(match,0)
locs1=np.zeros((num,2))
locs2=np.zeros((num,2))
   
for i in range(num):
   locs1[i]=key2[match[i,1]]
   locs2[i]=key1[match[i,0]]
    
locs1[:,[1,0]] = locs1[:,[0,1]] #swap coordinates 
locs2[:,[1,0]] = locs2[:,[0,1]] #swap coordinates
#compute homography
bestH2to1,inliers=computeHransac(locs1,locs2)


panoImage = imageStitching(img1,img2,bestH2to1)

skimage.io.imsave('pnc_pano.png',panoImage)
plt.subplot(1,2,1)
plt.imshow(img1)
plt.title('pnc0')
plt.subplot(1,2,2)
plt.title('pnc1')
plt.imshow(img2)
plt.figure()
plt.imshow(panoImage)
plt.show()

# Q 4.2
panoImage2= imageStitching_noClip(img1,img2,bestH2to1)
skimage.io.imsave('incline_pano.jpg',panoImage2)
plt.subplot(2,1,1)
plt.imshow(panoImage)
plt.subplot(2,1,2)
plt.imshow(panoImage2)
plt.show()

# Q 4.3
panoImage3 = generatePanorama(img1, img2)

# Q 4.4 (EC)
# Stitch your own photos with your code

# Q 4.5 (EC)
# Write code to stitch multiple photos
# see http://www.cs.jhu.edu/~misha/Code/DMG/PNC3/PNC3.zip
# for the full PNC dataset if you want to use that
if False:
    imgs = [skimage.io.imread('../PNC3/src_000{}.png'.format(i)) for i in range(7)]
    panoImage4 = generateMultiPanorama(imgs)
    plt.imshow(panoImage4)
    plt.show()