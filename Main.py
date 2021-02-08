import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
import time
import Model
import Get_data
def main(X, y, std, func):
    start_time = time.time()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)
    model = Model.PNN(X_train, y_train, X_test, y_test, std, func)
    y_predicted = model.predict()
    score = metrics.accuracy_score(y_test, y_predicted)
    print("Accuracy: ", score)
    ber = 1 - score
    print("BER: ", ber)
    cf = metrics.confusion_matrix(y_test, y_predicted.astype(int))
    print("Confusion matrix: ", cf)
    print("--- %s seconds ---" % (time.time() - start_time))
    return score
if __name__ == '__main__':
    distance = st.sidebar.selectbox("Choose a distance (cm)", ("10", "20", "30"))
    bit_rate = st.sidebar.selectbox("Choose an bit rate (kbps)", ("200", "250", "300"))
    model_type = st.sidebar.selectbox("Choose an model", ("PNN", "GRNN"))
    if model_type == "PNN":
        st.title('PNN model')
    else: 
        st.title('GRNN model')
    std = st.slider("Choose a standard deviation: ", min_value = 0.01, max_value= 1.0, value= 0.01, step=0.01)
    func = st.selectbox("Choose an activation function", ("1", "2", "3"))
    distance = 20
    bit_rate = 250
    X, y = Get_data.get_data(distance, bit_rate)
    X_temp = X.reshape(-1)
    y_temp = np.kron(y, [1, 1, 1, 1, 1])
    df = np.vstack([X_temp, y_temp])
    df = np.transpose(df)
    df = pd.DataFrame(df, columns = ['Transmitted signal', 'Received signal'])
    st.line_chart(df[:200])
    score = main(X, y, std, func) 
    st.write("Accuracy: ", score)