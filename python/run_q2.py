from q2 import makeTestPattern, computeBrief, briefLite, briefMatch, testMatch, briefRotTest, briefRotLite,plotMatches

import scipy.io as sio
import skimage.color
import skimage.io
import skimage.feature

# Q2.1
compareX, compareY = makeTestPattern(9,256)
sio.savemat('testPattern.mat',{'compareX':compareX,'compareY':compareY})

# Q2.2
img1 = skimage.io.imread('../data/model_chickenbroth.jpg')
im1 = skimage.color.rgb2gray(img1)
img2 = skimage.io.imread('../data/model_chickenbroth.jpg')
im2 = skimage.color.rgb2gray(img2)

# YOUR CODE: Run a keypoint detector, with nonmaximum supression
# locs holds those locations n x 2
#locs1 = None
#locs2 = None

locs1, desc1 = computeBrief(im1,locs1,compareX,compareY,patchWidth=9)
locs2, desc2 = computeBrief(im2,locs2,compareX,compareY,patchWidth=9)
# Q2.3
locs1, desc1 = briefLite(im1)
locs2, desc2 = briefLite(im2)
# Q2.4

matches = briefMatch(desc1,desc2,ratio=0.8)
plotMatches(im1,im2,matches,locs1,locs2)

testMatch()

# Q2.5
briefRotTest()

# EC 1
briefRotTest(briefRotLite)

# EC 2 
# write it yourself!