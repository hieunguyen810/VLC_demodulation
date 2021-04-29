import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
import time
import Model
import Get_data
def main(X, y, std, model_type):
    start_time = time.time()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)
    if model_type == "PNN":
        model = Model.PNN(X_train, y_train, X_test, y_test, std)
        y_predicted = model.predict()
    else:
        model = Model.GRNN(X_train, y_train, X_test, y_test, std)
        y_predicted = model.predict()
        y_predicted = y_predicted.astype(int)
    score = metrics.accuracy_score(y_test, y_predicted)
    print("Accuracy: ", score)
    ber = 1 - score
    print("BER: ", ber)
    cf = metrics.confusion_matrix(y_test, y_predicted.astype(int))
    print("Confusion matrix: ", cf)
    print("--- %s seconds ---" % (time.time() - start_time))
    return score
if __name__ == '__main__':
    mode = st.sidebar.selectbox("Choose a preprocessing mode: ", ("normal", "Framing", "Continuous wavelet transform", "Denoising autoencoder"))
    bit_rate = st.slider("Choose an bit rate (Kbps): ", min_value = 50, max_value= 400, value= 100, step=50)
    model_type = st.sidebar.selectbox("Choose an model", ("PNN", "GRNN"))
    if model_type == "PNN":
        st.title('PNN model')
        std = st.slider("Choose a standard deviation: ", min_value = 0.01, max_value= 1.0, value= 0.01, step=0.01)
    else: 
        st.title('GRNN model')
        std = st.slider("Choose a standard deviation: ", min_value = 0.01, max_value= 5, value= 0.01, step=0.05)
    distance = 20
    #bit_rate = 250
    X1, X2, y = Get_data.get_data(distance, bit_rate, mode)
    X_temp = X1.reshape(-1)
    y_temp = np.kron(y, [1, 1, 1, 1, 1])
    df = np.vstack([X_temp, y_temp])
    df = np.transpose(df)
    df = pd.DataFrame(df, columns = ['Transmitted signal', 'Received signal'])
    st.line_chart(df[:200])
    score = main(X2, y, std, model_type) 
    st.write("Accuracy: ", score)
    st.write("BER: ", 1 - score)
