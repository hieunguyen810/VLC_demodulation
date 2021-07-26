import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
def read_data(filename):
    data = pd.read_csv(filename, header=None)
    X = data.values
    return X
def read_label(filename):
    label = pd.read_csv(filename, header = None)
    y = label.values
    y = y.reshape(-1)
    return y
def test_each_file(bit_rate, num_file):
    j = 1
    while j < (num_file+1):
        X_path = 'Lorentz_dataset/10cm/%sk/%s000_250_%s.csv' % (bit_rate, bit_rate, j)
        y_path = 'Lorentz_dataset/10cm/%sk/label_%s000_250_%s.csv' % (bit_rate, bit_rate, j)
        X = read_data(X_path)
        y = read_label(y_path)
        y = y.reshape(-1)
        data = []
        for i in np.arange(len(X)):
            if i == 0:
                temp_1 = X[len(X)-1]
                temp_2 = X[i]
                temp_3  = X[i+1]
                temp_4 = np.hstack([temp_1, temp_2, temp_3])
            elif i == len(X)-1:
                temp_1 = X[i-1]
                temp_2 = X[i]
                temp_3 = X[0]
                temp_4 = np.hstack([temp_1, temp_2, temp_3])
            else:
                temp_1 = X[i-1]
                temp_2 = X[i]
                temp_3 = X[i+1]
                temp_4 = np.hstack([temp_1, temp_2, temp_3])
            data = np.append(data, temp_4)
        data = data.reshape([250, 15])
        X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.33, random_state=42)
        model = LogisticRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_pred, y_test)
        print("Accuracy: ", j, acc)
        j+=1
    return 
def feature_importance(bit_rate):
    filename_1 = "Lorentz_dataset/Lorentz_20cm_%sk_related_bit.csv" % bit_rate
    filename_2 = "Lorentz_dataset/Lorentz_label_20cm_%sk.csv" % bit_rate
    X = pd.read_csv(filename_1, header = None)
    X = X.values
    y = pd.read_csv(filename_2, header = None)
    y = y.values
    y = y.reshape([y.shape[0], ])
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor()
    # model = LogisticRegression()
    model.fit(X_train, y_train)
    importance = model.feature_importances_
    #importance = model.coef_[0] # logistic regression
    # from scipy import stats
    # importance = stats.zscore(importance)
    plt.bar([x for x in np.arange(0, X.shape[1], 1)], importance)
    plt.xlabel("Features")
    plt.ylabel("Score")
    # plt.plot(importance)
    plt.show()
def main():
    # test_each_file(100, 42)
    feature_importance(250)
if __name__ == '__main__':
    main() 
