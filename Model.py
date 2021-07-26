import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from numpy.core.umath_tests import inner1d
import math
from sklearn import metrics
import tensorflow as tf
import random
from sklearn.preprocessing import MinMaxScaler
from os import path
import scipy as sp
class GRNN:
    def __init__(self, x_train, y_train, x_test, y_test, std):
        self.x_train= x_train
        self.y_train= y_train
        self.x_test= x_test
        self.y_test= y_test
        self.std = np.full((1, self.y_train.size), std)
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
            predict_array = predict_array.astype(int)
        return predict_array*2
class PNN:
    def __init__(self, x_train, y_train, x_test, y_test, std):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.std = std
    def activation_func(self, distances): #kernel       
        kernel = np.exp((-distances)/(2*self.std**2))   # for dot product
        return np.sum(kernel)
    def output(self, my_class, i):
        distances = np.subtract(my_class, self.x_test[i])
        # distances = distances * signal.windows.triang(25)
        distances = inner1d(distances, distances) #dot product 
        return self.activation_func(distances)
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
class DAE:
    def __init__(self, bit_rate, X, y):
        self.bit_rate = bit_rate
        self.num = 1500
        self.z_dim = 5
        self.noise_level = 0.2
        self.X_test = X
        self.y = y
    def scale(self, X):
        s = MinMaxScaler()
        X = X.reshape(20*self.z_dim, 1)
        X = s.fit_transform(X)
        X = X.reshape(20, 1, self.z_dim)
        return X
    def getDeepAE(self):
        # input layer
        input_layer = tf.keras.layers.Input(shape=(1, self.z_dim))
        encode_layer = tf.keras.layers.LSTM(20, activation='relu', return_sequences = True, kernel_regularizer=tf.keras.regularizers.L1(0.0))(input_layer)
        encode_layer = tf.keras.layers.LSTM(self.z_dim, activation='relu', return_sequences = False)(encode_layer)
        latent_view = tf.keras.layers.RepeatVector(self.z_dim)(encode_layer)
        decode_layer = tf.keras.layers.LSTM(self.z_dim, activation='relu', return_sequences = True)(latent_view)
        decode_layer = tf.keras.layers.LSTM(20, activation='relu', return_sequences = True)(decode_layer)
        # decode_layer = tf.keras.layers.Dropout(0.2)(decode_layer)
        output_layer = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1))(decode_layer)
        # model
        model = tf.keras.Model(input_layer, output_layer)
        return model
    def correct(self):
        autoencoder = self.getDeepAE()
        optimizer = tf.keras.optimizers.SGD(lr=0.001, momentum = 0.9, nesterov=False)
        loss = tf.keras.losses.MeanSquaredError()
        autoencoder.compile(optimizer = optimizer, loss = loss)
        self.X_test = self.X_test.reshape([self.X_test.shape[0]*self.z_dim, 1])
        scale_1 = MinMaxScaler()
        self.X_test = scale_1.fit_transform(self.X_test)
        self.X_test = self.X_test.reshape([int(self.X_test.shape[0]/self.z_dim), 1, self.z_dim])
        bit_0 = np.zeros(self.z_dim)
        bit_1 = np.full((self.z_dim, ), 1)
        X_train = []
        for i in range(self.num):
            a = random.randint(0, 1)
            if a == 0:
                X_train = np.append(X_train, bit_0 + np.random.randn()/5)
            else:
                X_train = np.append(X_train, bit_1 + np.random.randn()/5)
        noise = np.random.normal(0, self.noise_level, self.num*self.z_dim)
        X_train_noisy = X_train + noise
        X_train_noisy = np.reshape(X_train_noisy, [self.num, 1, self.z_dim])
        X_train = np.reshape(X_train, [self.num, 1, self.z_dim])
        X_val = self.X_test[:20]
        X_val = X_val.reshape(X_val.shape[0], 1, self.z_dim)
        X_val_true = np.kron(self.y[:20], np.full((self.z_dim, ), 1))
        X_val_true = X_val_true.reshape(20, 1, self.z_dim)
        if path.exists("model_ae.h5"):
            h = autoencoder.load_weights("model_ae.h5")
        else:
            h = autoencoder.fit(X_train_noisy, X_train, epochs=100, batch_size=5, shuffle=True, validation_data=(self.scale(X_val), X_val_true), verbose = 0)
            autoencoder.save("model_ae.h5")
        X_pred = autoencoder.predict(self.X_test)
        X_pred = X_pred.reshape(-1)
        b, a = sp.signal.butter(3, 0.6)
        filt = sp.signal.filtfilt(b, a, X_pred)
        filt = filt.reshape([int(self.X_test.shape[0]/self.z_dim), self.z_dim])
        return filt
class DTNN:
    def __init__(self, x_train, y_train, x_test, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
    def DTNN_model(self):
        inputs = tf.keras.Input(shape = (2, 75))
        Layer1 = tf.keras.layers.Dense(75, activation = 'relu')(inputs)
        split0, split1 = tf.split(Layer1, num_or_size_splits = 2, axis = 1)
        H1 = tf.keras.layers.Dense(3, activation='relu')(split0)
        H2 = tf.keras.layers.Dense(3, activation='relu')(split1)
        V = tf.linalg.cross(H1, H2)
        Layer2 = tf.keras.layers.Dense(10, activation = 'relu')(V)
        outputs = tf.keras.layers.Dense(2, activation = 'sigmoid')(Layer2)
        model = tf.keras.Model(inputs = inputs, outputs = outputs)
        return model
    def predict(self):
        num_epochs = 25
        optimizers = tf.keras.optimizers.SGD(learning_rate=0.001)
        loss = tf.keras.losses.SparseCategoricalCrossentropy()
        model = self.DTNN_model()
        model.compile(loss = loss, optimizer = optimizers, metrics = ['accuracy'])
        x_train, x_val, y_train, y_val = train_test_split(self.x_train, self.y_train, test_size=0.2, random_state=42)
        h = model.fit(x_train, y_train, validation_data =(x_val, y_val), batch_size=25, epochs=num_epochs, verbose=0)
        score = model.evaluate(self.x_test, self.y_test, verbose = 0)
        return score[1]