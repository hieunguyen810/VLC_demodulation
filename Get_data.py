import pandas as pd
def get_data(distance, bit_rate):
    file_name_1 = "Full_dataset/Lorentz/%scm/Lorentz_%scm_%sk.csv" % (distance, distance, bit_rate) 
    file_name_2 = "Full_dataset/Lorentz/%scm/Lorentz_label_%scm_%sk.csv" % (distance, distance, bit_rate) 
    X = pd.read_csv(file_name_1, header = None)
    X = X.values
    y = pd.read_csv(file_name_2, header = None)
    y = y.values
    y = y.reshape([len(y), ])
    return X, y