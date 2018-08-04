import math 
import numpy as np     
import cv2 as cv                        #opencv
import os
from sklearn import svm
from sklearn.externals import joblib    #model load\archive lib

def main():

    #HOG descriptor
    winSize = (64,128)
    blockSize = (16,16)
    blockStride = (8,8)
    cellSize = (8,8)
    nbins = 9
    global hog 
    hog = cv.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins)

    #define HOG computation
    global winStride
    winStride = (16,16)
    global padding
    padding = (0,0)
    train()
    
    print ("The accuracy on train_data is",test_traindata())
    print ("The accuracy on normal test data is",test_normal())
    print ("The accuracy on normal people in haunted house is",test_normal2())
    print ("The accuracy on horror image is",test_novelty())

    
def train():
    path='C:\\Users\\wangy\\Desktop\\S1S2\\EZ_media\\data\\traindata'
    files= os.listdir(path)			#read name of data image


    #define HOG computation
    winStride = (16,16)
    padding = (0,0)
    #define matrix to store histograms of all train data
    hist=np.ndarray((len(files),len(hog.compute(cv.resize(cv.imread(path+'\\'+files[0]),(599,536),interpolation=cv.INTER_NEAREST),winStride,padding))),dtype=np.float32)
    i=0
    print("calculating HOG vectors")
    for file_name in files:
        image=cv.imread(path+'\\'+file_name)
        image=cv.resize(image,(599,536),interpolation=cv.INTER_NEAREST)
        temp=hog.compute(image,winStride,padding)
        temp=temp.reshape(1,temp.shape[0])
        hist[i][:] =temp
        i+=1
    print("HOG vectors calculation complete")

    #train One Class_SVM
    clf = svm.OneClassSVM(nu=0.003, kernel="linear")
    print("fitting model")
    clf.fit(hist)
    print("model fitting complete")

    #archive trained model
    print("archive trained model")
    joblib.dump(clf,'C:\\Users\\wangy\\Desktop\\S1S2\\EZ_media\\model_save\\clf_linear_95_0.003.pkl')
    print("trained model archived")
    #load trained model from pkl file
    #clf3=joblib.load('C:\\Users\\wangy\\Desktop\\S1S2\\EZ_media\\model_save\\clf_linear_95_nu=0.003.pkl')


def test_traindata():
    path='C:\\Users\\wangy\\Desktop\\S1S2\\EZ_media\\data\\traindata'
    files= os.listdir(path)			#read name of data image

    clf=joblib.load('C:\\Users\\wangy\\Desktop\\S1S2\\EZ_media\\model_save\\clf_linear_95_nu=0.003.pkl')   #read model
    #define matrix to store histograms of all train data
    hist=np.ndarray((len(files),len(hog.compute(cv.resize(cv.imread(path+'\\'+files[0]),(599,536),interpolation=cv.INTER_NEAREST),winStride,padding))),dtype=np.float32)
    i=0
    print("calculating HOG vectors for train_data")
    for file_name in files:
        image=cv.imread(path+'\\'+file_name)
        image=cv.resize(image,(599,536),interpolation=cv.INTER_NEAREST)
        temp=hog.compute(image,winStride,padding)
        temp=temp.reshape(1,temp.shape[0])
        hist[i][:] =temp
        i+=1
    print("HOG vectors calculation complete")

    calc=0
    print("predicting trained data")
    predict=clf.predict(hist)
    total=hist.shape[:][0]
    for i in range(total):
        if predict[i]==1 :
            calc+=1
    return(calc/total)



def test_normal():
    path='C:\\Users\\wangy\\Desktop\\S1S2\\EZ_media\\data\\testdata_normal'
    files= os.listdir(path)			#read name of data image

    clf=joblib.load('C:\\Users\\wangy\\Desktop\\S1S2\\EZ_media\\model_save\\clf_linear_95_nu=0.003.pkl')   #read model
    #define matrix to store histograms of all train data
    hist=np.ndarray((len(files),len(hog.compute(cv.resize(cv.imread(path+'\\'+files[0]),(599,536),interpolation=cv.INTER_NEAREST),winStride,padding))),dtype=np.float32)
    i=0
    print("calculating HOG vectors for test_data_normal")
    for file_name in files:
        image=cv.imread(path+'\\'+file_name)
        image=cv.resize(image,(599,536),interpolation=cv.INTER_NEAREST)
        temp=hog.compute(image,winStride,padding)
        temp=temp.reshape(1,temp.shape[0])
        hist[i][:] =temp
        i+=1
    print("HOG vectors calculation complete")

    calc=0
    print("predicting test_data_normal")
    predict=clf.predict(hist)
    total=hist.shape[:][0]
    for i in range(total):
        if predict[i]==1 :
            calc+=1
    return(calc/total)



def test_normal2():
    path='C:\\Users\\wangy\\Desktop\\S1S2\\EZ_media\\data\\people_in_haunted_house'
    files= os.listdir(path)			#read name of data image

    clf=joblib.load('C:\\Users\\wangy\\Desktop\\S1S2\\EZ_media\\model_save\\clf_linear_95_nu=0.003.pkl')   #read model
    #define matrix to store histograms of all train data
    hist=np.ndarray((len(files),len(hog.compute(cv.resize(cv.imread(path+'\\'+files[0]),(599,536),interpolation=cv.INTER_NEAREST),winStride,padding))),dtype=np.float32)
    i=0
    print("calculating HOG vectors for test_data_normal2")
    for file_name in files:
        image=cv.imread(path+'\\'+file_name)
        image=cv.resize(image,(599,536),interpolation=cv.INTER_NEAREST)
        temp=hog.compute(image,winStride,padding)
        temp=temp.reshape(1,temp.shape[0])
        hist[i][:] =temp
        i+=1
    print("HOG vectors calculation complete")

    calc=0
    print("predicting test_data_normal2")
    predict=clf.predict(hist)
    total=hist.shape[:][0]
    for i in range(total):
        if predict[i]==1 :
            calc+=1
    return(calc/total)


def test_novelty():
    path='C:\\Users\\wangy\\Desktop\\S1S2\\EZ_media\\data\\test_data_novelty'
    files= os.listdir(path)			#read name of data image

    clf=joblib.load('C:\\Users\\wangy\\Desktop\\S1S2\\EZ_media\\model_save\\clf_linear_95_nu=0.003.pkl')   #read model
    #define matrix to store histograms of all train data
    hist=np.ndarray((len(files),len(hog.compute(cv.resize(cv.imread(path+'\\'+files[0]),(599,536),interpolation=cv.INTER_NEAREST),winStride,padding))),dtype=np.float32)
    i=0
    print("calculating HOG vectors for test_data_novelty")
    for file_name in files:
        image=cv.imread(path+'\\'+file_name)
        image=cv.resize(image,(599,536),interpolation=cv.INTER_NEAREST)
        temp=hog.compute(image,winStride,padding)
        temp=temp.reshape(1,temp.shape[0])
        hist[i][:] =temp
        i+=1
    print("HOG vectors calculation complete")

    calc=0
    print("predicting test_data_novelty")
    predict=clf.predict(hist)
    total=hist.shape[:][0]
    for i in range(total):
        if predict[i]==-1 :
            calc+=1
    return(calc/total)



main()

