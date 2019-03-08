import numpy as np
from q2 import *
from q3 import *
import skimage.color
import skimage.io
import skimage
import scipy
from skimage.feature import ORB,match_descriptors
# you may find this useful in removing borders
# from pnc series images (which are RGBA)
# and have boundary regions
def clip_alpha(img):
    img[:,:,3][np.where(img[:,:,3] < 1.0)] = 0.0
    return img 

# Q 4.1
# this should be a less hacky version of
# composite image from Q3
# img1 and img2 are RGB-A
# warp order 0 might help
# try warping both images then combining
# feel free to hardcode output size
def imageStitching(img1, img2, H2to1):
    panoImg = None
    # YOUR CODE HERE
    H2to1_inv=linalg.inv(H2to1)
    matrix = np.array([[1, 0, 0], [0, 1, -10], [0, 0, 1]])
    warpImg1=skimage.transform.warp(img1, matrix,output_shape=(360,350), order=2, mode='constant')
    warpImg2=skimage.transform.warp(img2,matrix@H2to1,output_shape=warpImg1.shape, order=2, mode='constant')
   
    panoImg = np.maximum(warpImg1, warpImg2)
   
    return panoImg


# Q 4.2
# you should make the whole image fit in that width
# python may be inv(T) compared to MATLAB
def imageStitching_noClip(img1, img2, H2to1):
    panoImg = None
    # YOUR CODE HERE
    H1= np.shape(img1)[0]
    W1= np.shape(img1)[1]
    H2= np.shape(img2)[0]
    W2= np.shape(img2)[1]
    out_size=[900,1800]
    
    #define the corners of img2
    corner = np.array([[0,0,1],[0, W2, 1],[ H2,0, 1],[ H2, W2, 1]])
    # warping on the corners
    warp_c = H2to1@corner.T
    warp_c = warp_c/warp_c[2,:]
    translate_width = np.min(warp_c[0,:])
    width = np.max(warp_c[0,:]) - translate_width
    translate_height = np.min(warp_c[1,:])
    height = np.max(warp_c[1,:]) - translate_height

    scalar = out_size[1] /width;
    #out_size = [900, out_size[1]];
    
   
    H2to1_inv=linalg.inv(H2to1)
    matrix = np.array([[1, 0, 0], [0, 1, -163], [0, 0, 1]])
    
    warpImg1=skimage.transform.warp(img1, matrix,output_shape=(out_size[0],out_size[1]), order=0)
    warpImg2=skimage.transform.warp(img2,  matrix@H2to1,output_shape=warpImg1.shape, order=0)
    
    mask_img2 = np.zeros([np.size(img2,0), np.size(img2,1), np.size(img2,2)])
    mask_img2[0,:] = 1; mask_img2[-1,:] = 1; mask_img2[:,0] = 1; mask_img2[:,-1] = 1
    mask_img2 = scipy.ndimage.distance_transform_edt(mask_img2==0)
    mask_img2 = mask_img2/np.max(mask_img2[:])
    mask2_warped = skimage.transform.warp(mask_img2, matrix@H2to1,output_shape=(out_size[0],out_size[1]))
    result2 = warpImg2 * mask2_warped
    
    mask_img1 = np.zeros([np.size(img1,0), np.size(img1,1), np.size(img1,2)])
    mask_img1[0,:] = 1; mask_img1[-1,:] = 1; mask_img1[:,0] = 1; mask_img1[:,-1] = 1
    mask_img1 = scipy.ndimage.distance_transform_edt(mask_img1==0)
    mask_img1 = mask_img1/np.max(mask_img1[:])
    mask1_warped = skimage.transform.warp(mask_img1, matrix,output_shape=(out_size[0],out_size[1]))
    result1 = warpImg1 * mask1_warped
    
    panoImg = (result1 + result2) / (mask1_warped + mask2_warped)
    panoImg.astype(np.float)
    #panoImg = np.maximum(warpImg1, warpImg2)
    skimage.io.imshow(panoImg)
    return panoImg

# Q 4.3
# should return a stitched image
# if x & y get flipped, np.flip(_,1) can help
def generatePanorama(img1, img2):
    panoImage = None
    # YOUR CODE HERE
    img1 = skimage.io.imread('../data/incline_L.png')
    img2 = skimage.io.imread('../data/incline_R.png')
    #skimage.io.imshow(img1)
    #skimage.io.imshow(img2)
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
    panoImage= imageStitching_noClip(img1,img2,bestH2to1)
    return panoImage

# Q 4.5
# I found it easier to just write a new function
# pairwise stitching from right to left worked for me!
def generateMultiPanorama(imgs):
    panoImage = None
    # YOUR CODE HERE
    
    return panoImage
