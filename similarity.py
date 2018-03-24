from HOG import HOG
import matplotlib.pyplot as plt
import numpy as np


def euclidian_metric(x,y):
    return np.linalg.norm(x-y)

def cosine_similarity(x,y):
    norm = np.linalg.norm(x) * np.linalg.norm(y)

    if norm == 0:
        return 0
    else:
        return np.dot(x,np.transpose(y)) / norm


def similarity(img1, img2, w, h, nb_bins, metric):
    '''Computes similarity between image 1 and image 2. Image 1 is taken has reference'''

    hog_similar_1 = HOG(img1,w,h,nb_bins,False)
    hog_similar_2 = HOG(img2,w,h,nb_bins,False)

    I,J,k = np.shape(hog_similar_1)  #Object 1 is used as reference

    metric_values = np.zeros((I,J))


    for i in range(I):
        for j in range(J):
            metric_values[i,j] = metric(hog_similar_1[i][j], hog_similar_2[i][j])

    result = np.mean(metric_values)

    return result


if __name__=="__main__":

    from skimage import io

    img_similar = io.imread('HOG 2/hog_similar.bmp')
    img_similar_1 = img_similar[22:,0:64]
    img_similar_2 = img_similar[22:,90:154]

    img_different = io.imread('HOG 2/hog_different.bmp')
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

    #--Ploting--

    # fig = plt.figure()
    # ax1 = fig.add_subplot(131)
    # ax2 = fig.add_subplot(132) 
    # ax3 = fig.add_subplot(133) 

    # ax1.imshow(img_similar)
    # ax2.imshow(img_similar_1)
    # ax3.imshow(img_similar_2)

    # plt.show()