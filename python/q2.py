import numpy as np
import scipy.io as sio
import scipy
import skimage.feature
import matplotlib.pyplot as plt
from skimage.feature import (corner_peaks, corner_harris,corner_fast)
# Q2.1
# create a 2 x nbits sampling of integers from to to patchWidth^2
# read BRIEF paper for different sampling methods
def makeTestPattern(patchWidth, nbits):
    res = None
    # YOUR CODE HERE
    compareX = np.zeros((nbits,1))
    compareY = np.zeros((nbits,1))
    
    for i in range(nbits):
        tempX = np.random.uniform(low=0.0, high=patchWidth**2-1, size = nbits)
        tempY = np.random.uniform(low=0.0, high=patchWidth**2-1, size = nbits)
    
    compareX = np.array(np.floor(tempX))
    compareY = np.array(np.floor(tempY))
    
    compareX=compareX.astype(int)
    compareY=compareY.astype(int)
   
    res = np.array([compareX,compareY])
    sio.savemat('testPattern.mat', {'compareX':compareX,'compareY':compareY})
    #print(res)
    return res 
#makeTestPattern(9, 256)

# Q2.2
# im is 1 channel image, locs are locations
# compareX and compareY are idx in patchWidth^2
# should return the new set of locs and their descs
def computeBrief(im,locs,compareX,compareY,patchWidth=9):
    desc = None
    # YOUR CODE HERE
    height = np.shape(im)[0]
    width = np.shape(im)[1]
    halfPatch = (patchWidth-1)/2
    #im = np.array(im)
    
    locs = locs[locs[:,0] >= 4,:]
    locs = locs[locs[:,0] < width - 4,:]
    locs = locs[locs[:,1] >= 4,:]
    locs = locs[locs[:,1] < height - 4,:]
    #print(locs) 
   
    m = locs.shape[0]
    nbits = len(compareX)
    desc = np.zeros((m, 256))
    xIdx =np.zeros((1,256))
    yIdx =np.zeros((1,256))
    xcol =np.zeros((1,256))
    xrow =np.zeros((1,256))
    ycol =np.zeros((1,256)) 
    yrow =np.zeros((1,256))
    for i in range(m):
        x = locs[i,0]
        y = locs[i,1]
        
        # get 9*9 matris from image
        patch = im[x-4:x+5][:,y-4:y+5]
        #print(np.shape(patch))
        
        for j in range(256):
            xIdx = compareX
            yIdx = compareY
            
            xcol = (xIdx%9).astype(int)
            xrow = ((xIdx-xcol)/9).astype(int)
            ycol = (yIdx%9).astype(int)
            yrow = ((yIdx-ycol)/9).astype(int)
            #print(xcol,xrow)
            if (patch[xrow[0,j]-1,xcol[0,j]-1] < patch[yrow[0,j]-1,ycol[0,j]-1]).any():
                desc[i,j] = 1
     
    return locs, desc

# Q2.3
# im is a 1 channel image
# locs are locations
# descs are descriptors
# if using Harris corners, use a sigma of 1.5
def briefLite(im):
    locs, desc = None, None
    # YOUR CODE HERE
    mat = sio.loadmat('testPattern.mat')
    compareX = mat['compareX']
    compareY = mat['compareY']
    patchWidth = 9
    #locs = corner_peaks(corner_fast(im, n=12, threshold=0.1))
    locs = corner_peaks(corner_harris(im,sigma = 1.5), min_distance=1)
    locs, desc =computeBrief(im,locs,compareX,compareY,patchWidth);
    
    return locs, desc


# Q 2.4
def briefMatch(desc1,desc2,ratio=0.8):
    # okay so we say we SAY we use the ratio test
    # which SIFT does
    # but come on, I (your humble TA), don't want to.
    # ensuring bijection is almost as good
    # maybe better
    # trust me
    matches = skimage.feature.match_descriptors(desc1,desc2,'hamming',cross_check=True)
    return matches

def plotMatches(im1,im2,matches,locs1,locs2):
    fig, ax = plt.subplots(nrows=1, ncols=1)
    skimage.feature.plot_matches(ax,im1,im2,locs1,locs2,matches,matches_color='r')
    plt.show()
    return

def testMatch():
    # YOUR CODE HERE
    img1 = skimage.io.imread('../data/model_chickenbroth.jpg')
    im1 = skimage.color.rgb2gray(img1)
    img2 = skimage.io.imread('../data/model_chickenbroth.jpg')
    im2 = skimage.color.rgb2gray(img2)
    
    locs1, desc1 = briefLite(im1)
    locs2, desc2 = briefLite(im2)
    
    matches = briefMatch(desc1, desc2)
    plotMatches(im1,im2,matches,locs1,locs2)
    
    return


# Q 2.5
# we're also going to use this to test our
# extra-credit rotational code
def briefRotTest(briefFunc=briefLite):
    # you'll want this
    import skimage.transform
    # YOUR CODE HERE
    img1 = skimage.io.imread('../data/model_chickenbroth.jpg')
    im1 = skimage.color.rgb2gray(img1)
    locs1, desc1 = briefLite(im1)
    for i in range(37):
        #tform = skimage.transform.SimilarityTransform(translation=(-10, 0))
        #im2_ = skimage.transform.warp(im1, tform)
        im2 = skimage.transform.rotate(im1,i*10)
        locs2, desc2 = briefLite(im2)
        matches = briefMatch(desc1,desc2,ratio=0.8)
        plotMatches(im1,im2,matches,locs1,locs2)
    return

# Q2.6
# YOUR CODE HERE


# put your rotationally invariant briefLite() function here
def briefRotLite(im):
    locs, desc = None, None
    # YOUR CODE HERE
    #Using Homography rotate
    img1 = skimage.io.imread('../data/model_chickenbroth.jpg')
    im1 = skimage.color.rgb2gray(img1)
    locs1, desc1 = briefLite(im1)
    for i in range(37):
        H_rot = [[cos(pi*i/360),  sin(pi*i/360), 0] 
                [-sin(pi*i/360),  cos(pi*i/360), 0]
                [ 0,              0,             1]]
        im2 = skimage.transform.rotate(im1,i*10);
        locs2, desc2 = briefLite(im2)
        matches = briefMatch(desc1,desc2,ratio=0.8)
        plotMatches(im1,im2,matches,locs1,locs2)
    return
    return locs, desc