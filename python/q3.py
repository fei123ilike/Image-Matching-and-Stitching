import numpy as np
import skimage.color
import skimage.io
from scipy import linalg



# Q 3.1
def computeH(l1, l2):
    H2to1 = np.eye(3)
    # YOUR CODE HERE
   
    N = np.shape(l1)[0]
    A =np.zeros((2*N,9))
    for i in range(N):
        x,y   = l2[i,0],l2[i,1]
        xp,yp = l1[i,0],l1[i,1]
        A[2*i+1, :] = [-x,-y,-1, 0, 0, 0,x*xp,y*xp,xp]  #odd rows
        A[2 * i, :] = [0, 0, 0,-x,-y,-1,x*yp,y*yp,yp]   #even rows
    
    u, sigma, vh = np.linalg.svd(A, full_matrices=True)  
    v = vh.T
    h = v[:,-1] 
    H2to1 = np.reshape(h,(3,3))
    
    return H2to1

# Q 3.2
def computeHnorm(l1, l2):
    H2to1 = np.eye(3)
    # YOUR CODE HERE
    
    N = np.shape(l1)[0]
    mean_l1_x = np.mean(l1[:,0])
    mean_l1_y = np.mean(l1[:,1])
    mean_l2_x = np.mean(l2[:,0])
    mean_l2_y = np.mean(l2[:,1])
    mean_l1 = [mean_l1_x, mean_l1_y]
    mean_l2 = [mean_l2_x, mean_l2_y]
    translated_l1 = l1 - mean_l1
    translated_l2 = l2 - mean_l2
    
    
    max_1_x = np.max(np.abs(translated_l1[:,0]))
    max_1_y = np.max(np.abs(translated_l1[:,1]))
    max_2_x = np.max(np.abs(translated_l2[:,0]))
    max_2_y = np.max(np.abs(translated_l2[:,1]))
    scale_1_x = np.sqrt(2)/max_1_x
    scale_1_y = np.sqrt(2)/max_1_y
    scale_2_x = np.sqrt(2)/max_2_x
    scale_2_y = np.sqrt(2)/max_2_y
    
    
    #Transform matrixes
    T1 = np.array([[scale_1_x,0,-scale_1_x*mean_l1_x],[0,scale_1_y,-scale_1_y*mean_l1_y],[0,0,1]])
    T2 = np.array([[scale_2_x,0,-scale_2_x*mean_l2_x],[0,scale_2_y,-scale_2_y*mean_l2_y],[0,0,1]])
    
    extra_ones = np.array(np.ones((N,1)))
    l1_dummy = (np.concatenate((l1, extra_ones),axis=1)).T
    l2_dummy = (np.concatenate((l2, extra_ones),axis=1)).T
    l1_norm = (T1@l1_dummy).T
    l2_norm = (T2@l2_dummy).T
    #print(np.shape(l1_dummy))
    #print(np.shape(l1_norm))
    H2to1_norm = computeH(l1_norm[:,:2], l2_norm[:,:2])
    #H2to1_norm = H2to1_norm/H2to1_norm[2,2]
    T1_inv = np.linalg.inv(T1)
    H2to1 = (T1_inv @ H2to1_norm) @ T2
    H2to1 = H2to1/H2to1[2,2]
    return H2to1

# Q 3.3
def computeHransac(locs1, locs2):
    bestH2to1, inliers = None, None
    # YOUR CODE HERE
    
    iteration=20000
    num=len(locs1)
    bestInliers=0
    inliers_max=np.zeros((1,num))
    
    
    for i in range(iteration):
        
        randomIdx = np.random.randint(0,num,4) 
        random_1 = np.array(locs1[randomIdx,:])
        random_2 = np.array(locs2[randomIdx,:])
        H2to1=computeHnorm(random_1,random_2)
        
        inliers=np.zeros((1,num))
        for i in range(num):
#            extra_ones = np.array(np.ones((num,1)))
#            l1_dummy = (np.concatenate((locs1, extra_ones),axis=1)).T
#            l2_dummy = (np.concatenate((locs2, extra_ones),axis=1)).T
#            l1_dummy.astype(np.int)
#            l2_dummy.astype(np.int)
#            l1_predict=(H2to1/H2to1[2,2])@l2_dummy
#            l1_predict=l1_predict/l1_predict[2][0]#divided x,y by z
#            diff=l1_dummy-l1_predict
            l2=np.array([[locs2[i][0]],[locs2[i][1]],[1]])
            l1=np.array([[locs1[i][0]],[locs1[i][1]],[1]])
            l1_predict=(H2to1/H2to1[2,2])@l2#calculate the predict points
            l1_predict=l1_predict/l1_predict[2][0]#divided x,y by z
            
            diff=l1-l1_predict

            if np.linalg.norm(diff[:,0]) < 4:
                inliers[0][i]=1
            else:
                inliers[0][i]=0
        numInliers=np.sum(inliers)
        #update number of inliers, store the biggest one
        if numInliers>bestInliers:
            bestInliers=numInliers
            inliers_max=inliers
    
    bestInliers=bestInliers.astype(np.int)
    p1_new=np.zeros((bestInliers,2))
    p2_new=np.zeros((bestInliers,2))
    count=0
    for i in range(num):
        if inliers_max[0][i]==1:
            count+=1
            p1_new[count-1,:]=locs1[i]
            p2_new[count-1,:]=locs2[i]
    bestH2to1=computeHnorm(p1_new,p2_new)
    inliers=inliers_max
    
    return bestH2to1, inliers


# Q3.4
# skimage.transform.warp will help
def compositeH( H2to1, template, img ):
    compositeimg = img
    # YOUR CODE HERE
    warpImg=skimage.transform.warp(img,H2to1,output_shape=template.shape)
    m1=np.size(warpImg,0)
    m2=np.size(warpImg,1)
    #replace black with the template image
    for i in range(m1):
        for j in range(m2):
            if warpImg[i,j]==0:
                warpImg[i,j]=template[i,j]
    
    compositeimg=warpImg
    
    return compositeimg


def HarryPotterize():
    # we use ORB descriptors but you can use something else
    from skimage.feature import ORB,match_descriptors
    # YOUR CODE HERE
    #read images
    img1 = skimage.io.imread('../data/cv_cover.jpg')#cv_cover
    im1  = skimage.color.rgb2gray(img1)
    img2 = skimage.io.imread('../data/cv_desk.png')#cv_desk
    im2  = skimage.color.rgb2gray(img2)
    img3 = skimage.io.imread('../data/hp_cover.jpg')
    im3  = skimage.color.rgb2gray(img3)#hp_cover
    
    #get descriptors for cv cover
    descriptor_extractor=ORB(n_keypoints=600)
    descriptor_extractor.detect_and_extract(im1)
    key1=descriptor_extractor.keypoints
    descriptor1 = descriptor_extractor.descriptors
    #get descriptors for cv desk
    descriptor_extractor=ORB(n_keypoints=600)
    descriptor_extractor.detect_and_extract(im2)
    key2=descriptor_extractor.keypoints
    descriptor2 = descriptor_extractor.descriptors
    #find match points for cv_cover and cv_desk
    match = match_descriptors(descriptor1, descriptor2,cross_check=True)
    
    num=np.size(match,0)
    locs1=np.zeros((num,2))#cv_desk
    locs2=np.zeros((num,2))#cv_cover
   
    for i in range(num):
        locs1[i]=key2[match[i,1]]
        locs2[i]=key1[match[i,0]]
    
    locs1[:,[1,0]] = locs1[:,[0,1]] #swap coordinates 
    locs2[:,[1,0]] = locs2[:,[0,1]] #swap coordinates
    #compute homography
    bestH2to1,inliers=computeHransac(locs1,locs2)
    bestH2to1_inv=linalg.inv(bestH2to1)
    
    im3_resize = skimage.transform.resize(im3,output_shape=im1.shape)
    output=compositeH(bestH2to1_inv/bestH2to1_inv[2,2],im2,im3_resize)
    
    skimage.io.imshow(output)
    print(bestH2to1)
    return

