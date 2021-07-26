import numpy as np
import pandas as pd
import json
import Getdata
import Model
from sklearn.model_selection import KFold
from sklearn import metrics
import time
def read_configure_file():
        #read configure file
    start_time = time.time()
    f = open("configure.json")
    configure = json.load(f)
    mode_1 = configure["preprocessing"]
    if mode_1 == "Framing":
        mode_1 = "related_bit"
    if configure["enableCWT"] == "true":
        mode_2 = "CWT"
    else:
        mode_2 = ""
    bit_rate = configure["bitrate"]
    distance = configure["distance"]
    model = configure["model"]
    f.close()
    return model, bit_rate, distance, mode_1, mode_2
def get_std(model):
    #get std: for PNN and GRNN
    if model == "PNN":
        f = open("default_std_PNN.json")
    else:
        f = open("default_std_GRNN.json")
    data = json.load(f)
    if bit_rate == 100 or bit_rate == 200:
        std = data[mode_1][str(bit_rate)][str(distance)]
    else:
        std = data[mode_1][str(bit_rate)]
    if mode_2 == "CWT":
        std = std*2
    f.close()
    return std
def get_data(model, bit_rate, distance, mode_1, mode_2):
    if model == "DTNN":
        try: 
            file_name_1 = "Lorentz_dataset/Lorentz_%scm_%sk_%s%s.csv" % (distance, bit_rate, "DAE", mode_2)
            file_name_2 = "Lorentz_dataset/Lorentz_%scm_%sk_%s%s.csv" % (distance, bit_rate, "related_bit", mode_2)
            file_name_3 = "Lorentz_dataset/Lorentz_label_%scm_%sk.csv" % (distance, bit_rate)
            X1 = pd.read_csv(file_name_1, header=None)
            X2 = pd.read_csv(file_name_2, header=None)
            y = pd.read_csv(file_name_3, header=None)
            X1 = X1.values
            X2 = X2.values
            y = y.values         
        except:
            X1, y = Getdata.exe("DAE", bit_rate, distance)
            X2, y = Getdata.exe("Framing", bit_rate, distance)
            X1 = Getdata.CWT(X1)
            X2 = Getdata.CWT(X2)
        X_1 = np.arange(0, X1.shape[0]*X1.shape[1], 1)
        X_2 = np.arange(0, X2.shape[0]*X2.shape[1], 1)
        X1 = X1.reshape(-1)
        X1 = np.interp(X_2, X_1, X1)
        X1 = X1.reshape([X2.shape[0], X2.shape[1]])
        X = np.hstack([X1, X2])
        X = X.reshape([X2.shape[0], 2, X2.shape[1]])  
    else: 
        try:
            filename_1 = "Lorentz_dataset/Lorentz_%scm_%sk_%s%s.csv" % (distance, bit_rate, mode_1, mode_2)
            filename_2 = "Lorentz_dataset/Lorentz_label_%scm_%sk.csv" % (distance, bit_rate)
            X = pd.read_csv(filename_1, header = None)
            y = pd.read_csv(filename_2, header = None)
            X = X.values
            y = y.values
        except:
            X, y = Getdata.exe(mode_1, bit_rate, distance)
            if mode_2 == "CWT":
                X = Getdata.CWT(X)
    y = y.reshape([y.shape[0], ])
    return X, y
def main():
    start_time = time.time()
    model, bit_rate, distance, mode_1, mode_2 = read_configure_file()
    if model == "PNN" or model == "GRNN":
        std = get_std(model)
    X, y = get_data(model, bit_rate, distance, mode_1, mode_2)
    print("----------------------")
    print("Bit_rate: ", bit_rate)
    print("Distance: ", distance)
    print("Model: ", model)
    print("Preprocessing: ", mode_1)
    if mode_2 == "CWT":
        print("Enable CWT: True")
    else:
        print("Enable CWT: False")
    print("----------------------")
    ber = []
    kf = KFold(n_splits=10)
    for train_index, test_index in kf.split(X):
        #print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        if model == "DTNN":
            m = Model.DTNN(X_train, y_train, X_test, y_test)
        elif model == "GRNN":
            m = Model.GRNN(X_train, y_train, X_test, y_test, std)
        else:
            m = Model.PNN(X_train, y_train, X_test, y_test, std)
        if model == "DTNN":
            score = m.predict()
        else:
            y_predicted = m.predict()
            score = metrics.accuracy_score(y_test, y_predicted)
        print("Accuracy: ", score)
        ber = np.append(ber, 1-score)
    print("--------------------")
    print("BER: ", np.mean(ber))
    print("--- %s seconds ---" % (time.time() - start_time))
if __name__ == '__main__':
    main() 