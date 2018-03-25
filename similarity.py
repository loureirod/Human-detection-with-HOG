from HOG import HOG
import matplotlib.pyplot as plt
import numpy as np


def euclidian_metric(x,y):
    '''Computes euclidian metric between x and y'''
    return np.linalg.norm(x-y)


def cosine_similarity(x,y):
    '''Computes cosine similarity between x and y'''
    
    norm = np.linalg.norm(x) * np.linalg.norm(y)

    if norm == 0:
        return 0
    else:
        return np.dot(x,np.transpose(y)) / norm


def similarity(img1, img2, w, h, nb_bins, metric):
    '''Computes similarity between image 1 and image 2 by applying the given metric \
    to HOG representations of image 1 and 2. HOG representations are computed over \
    cells with size w x h and with nb_bins bins. Returns a float in [0,1]'''


    # Computation of HOG representations of img1 and img2
    hog_similar_1 = HOG(img1,w,h,nb_bins,False)
    hog_similar_2 = HOG(img2,w,h,nb_bins,False)


    I,J,k = np.shape(hog_similar_1)
    metric_values = np.zeros((I,J))

    # Computation of similarity 
    for i in range(I):
        for j in range(J):
            metric_values[i,j] = metric(hog_similar_1[i][j], hog_similar_2[i][j])

    result = np.mean(metric_values)

    return result


# Examples

if __name__=="__main__":

    from skimage import io

    img_similar = io.imread('HOG 2/hog_similar2.bmp')
    img_similar_1 = img_similar[22:,0:64]
    img_similar_2 = img_similar[22:,90:154]

    img_different = io.imread('HOG 2/hog_different2.bmp')
    img_different_1 = img_different[22:,0:64]
    img_different_2 = img_different[22:,90:154]

    print("Similarity of hog_similar.bmp objects :")

    print("-> For w=64 , h=128 , nb_bins=12 ")
    print(similarity(img_similar_1,img_similar_2,64,128,12,cosine_similarity))

    print("-> For w=16 , h=16 , nb_bins=9 ")
    print(similarity(img_similar_1,img_similar_2,16,16,9,cosine_similarity))

    print("Similarity of hog_different.bmp objects :")

    print("-> For w=64 , h=128 , nb_bins=12 ")    
    print(similarity(img_different_1,img_different_2,64,128,12,cosine_similarity))

    print("-> For w=16 , h=16 , nb_bins=9 ")    
    print(similarity(img_different_1,img_different_2,16,16,9,cosine_similarity))
