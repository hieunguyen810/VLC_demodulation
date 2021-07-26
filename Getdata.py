import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import Model
import pywt
def read_data(filename):
    data = pd.read_csv(filename, header=None)
    X = data.values
    X = X.reshape([len(X)*5, 1])
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    return X
def read_label(filename):
    label = pd.read_csv(filename, header = None)
    y = label.values
    return y
def get_data(bit_rate, distance):
    i = 1
    X = []
    y = []
    num_file = 40
    while i < (num_file+1):
        X_path = 'Lorentz_dataset/%scm/%sk/%s000_250_%s.csv' % (distance, bit_rate, bit_rate, i)
        y_path = 'Lorentz_dataset/%scm/%sk/label_%s000_250_%s.csv' % (distance, bit_rate, bit_rate, i)
        X_t = read_data(X_path)
        y_t = read_label(y_path)
        X = np.append(X, X_t)
        y = np.append(y, y_t)
        i+=1
    X = X.reshape(num_file*250, 5)
    y = y.reshape(-1)
    return X, y
def get_data_related_bit(bit_rate, distance):
    j = 1
    X = []
    y = []
    num_file = 40
    while j < (num_file+1):
        data_path = 'Lorentz_dataset/%scm/%sk/%s000_250_%s.csv' % (distance, bit_rate, bit_rate, j)
        y_path = 'Lorentz_dataset/%scm/%sk/label_%s000_250_%s.csv' % (distance, bit_rate, bit_rate, j)
        data_t = read_data(data_path)
        y_t = read_label(y_path)
        data_t = data_t.reshape([250, 5])
        y_t = y_t.reshape(-1)
        data = []
        for i in np.arange(len(data_t)):
            if i == 0:
                temp_1 = data_t[len(data_t)-2]
                temp_2 = data_t[len(data_t)-1]
                temp_3 = data_t[i]
                temp_4  = data_t[i+1]
                temp_5  = data_t[i+2]
                temp_6 = np.hstack([temp_1, temp_2, temp_3, temp_4, temp_5])
            elif i == 1:
                temp_1 = data_t[len(data_t)-1]
                temp_2 = data_t[len(data_t)-1]
                temp_3 = data_t[i]
                temp_4 = data_t[i+1]
                temp_5 = data_t[i+2]
                temp_6 = np.hstack([temp_1, temp_2, temp_3, temp_4, temp_5])
            elif i == len(data_t)-2:
                temp_1 = data_t[i-2]
                temp_2 = data_t[i-1]
                temp_3 = data_t[i]
                temp_4 = data_t[i+1]
                temp_5 = data_t[0]
                temp_6 = np.hstack([temp_1, temp_2, temp_3, temp_4, temp_5])
            elif i == len(data_t)-1:
                temp_1 = data_t[i-2]
                temp_2 = data_t[i-1]
                temp_3 = data_t[i]
                temp_4 = data_t[0]
                temp_5 = data_t[1]
                temp_6 = np.hstack([temp_1, temp_2, temp_3, temp_4, temp_5])
            else:
                temp_1 = data_t[i-2]
                temp_2 = data_t[i-1]
                temp_3 = data_t[i]
                temp_4 = data_t[i+1]
                temp_5 = data_t[i+2]
                temp_6 = np.hstack([temp_1, temp_2, temp_3, temp_4, temp_5])
            data = np.append(data, temp_6)
        X = np.append(X, data)
        y = np.append(y, y_t)
        j+=1
    X = X.reshape(num_file*250, 25)
    return X, y
def CWT(X):
    cwt = []
    for i in X:
        coef, freqs = pywt.cwt(i, np.arange(1, 4), 'mexh')
        cwt = np.append(cwt, coef)
    cwt = cwt.reshape(10000, X.shape[0]*3)
    return cwt
def exe(mode, bit_rate, distance):
    if mode == "related_bit":
        X, y = get_data_related_bit(bit_rate, distance)
    elif mode == "DAE":
        X, y = get_data(bit_rate, distance)
        m = Model.DAE(bit_rate, X, y)
        X = m.correct()
    else:
        X, y = get_data(bit_rate, distance)
    filename_1 = "Lorentz_dataset/Lorentz_%scm_%sk_related_bit.csv" % (distance, bit_rate)
    filename_2 = "Lorentz_dataset/Lorentz_label_%scm_%sk.csv" % (distance, bit_rate)
    np.savetxt(filename_1, X, delimiter=",")
    np.savetxt(filename_2, y, delimiter=",")
    return X, y