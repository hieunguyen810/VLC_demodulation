import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import math
#import seaborn as sn
import matplotlib.pyplot as plt
import random
from numpy.core.umath_tests import inner1d
class PNN:
    def __init__(self, x_train, y_train, x_test, y_test, std):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.std = std
    def activation_func(self, distances, my_class): #kernel
        kernel = np.exp((-distances)/(self.std**2))   # for dot product
        return np.sum(kernel)
    def output(self,my_class, i):
        #distances = np.abs(self.x_test[i]-my_class) # manhattan distance  
        #distances = np.sqrt((self.x_test[i]-my_class)**2)  #euclidean distances 
        distances = my_class - self.x_test[i] 
        distances = inner1d(distances, distances) #dot product  
        return self.activation_func(distances, my_class)
    def pdf(self, my_class, i):
        pdf = (1/(math.sqrt(2*math.pi)*self.std))*(1/self.y_test.size)*(self.output(my_class, i))
        return pdf
    def predict(self):
        label_1 = 0
        label_0 = 0
        predict_array = []
        class_0 = []
        class_1 = []
        a = self.x_train.shape
        for j in np.arange(len(self.y_train)):
            if self.y_train[j] == 1:
                label_1+=1
                class_1 = np.append(class_1, self.x_train[j])
            else:
                label_0+=1
                class_0 = np.append(class_0, self.x_train[j])
        class_0 = np.reshape(class_0, [int(len(class_0)/a[1]), a[1]])
        class_1 = np.reshape(class_1, [int(len(class_1)/a[1]), a[1]])
        p_prior_0 = label_0 / len(self.y_train)
        p_prior_1 = label_1 / len(self.y_train)
        for i in range(self.y_test.size):
            c_0 = p_prior_0 * self.pdf(class_0, i)
            c_1 = p_prior_1 * self.pdf(class_1, i)
            c = np.argmax(np.array([c_0, c_1]))
            predict_array = np.append(predict_array, c)
        return predict_array
class GRNN:
    def __init__(self,x_train,y_train,x_test,y_test):
        self.x_train= x_train
        self.y_train= y_train
        self.x_test= x_test
        self.y_test= y_test
        self.std = np.full((1, self.y_train.size), 3)
        # self.std = np.ones((1,self.y_train.size))#np.random.rand(1,self.train_y.size) #Standard deviations(std) are sometimes called RBF widths.
    def activation_func(self, distances): # gaussian kernel       
        kernel = np.exp(- (distances) / 2*(self.std**2))
        return kernel
    def output(self,i):#sometimes called weight
        distances = np.sqrt(np.sum((self.x_test[i]-self.x_train)**2,axis=1))# euclidean distance       
        return self.activation_func(distances)   
    def denominator(self,i):
        return np.sum(self.output(i))
    def numerator(self,i): 
        return np.sum(self.output(i) * self.y_train)   
    def predict(self):
        predict_array = []
        for i in range(self.y_test.size):
            if self.numerator(i) == 0 and self.denominator(i) == 0:
                predict = 1
            else:
                predict=np.array([self.numerator(i)/self.denominator(i)])
            predict_array=np.append(predict_array,predict)
        return predict_array*1.95

