import pandas as pd
def get_data(distance, bit_rate, mode):
    if mode == "normal":
        file_name_1 = "Full_dataset/Lorentz/%scm/Lorentz_%scm_%sk.csv" % (distance, distance, bit_rate) 
        file_name_2 = "Full_dataset/Lorentz/%scm/Lorentz_label_%scm_%sk.csv" % (distance, distance, bit_rate) 
        X1 = pd.read_csv(file_name_1, header = None)
        X1 = X1.values
        X2 = X1
        y = pd.read_csv(file_name_2, header = None)
        y = y.values
        y = y.reshape([len(y), ])
    else: 
        file_name_1 = "Full_dataset/Lorentz/%scm/Lorentz_%scm_%sk.csv" % (distance, distance, bit_rate) 
        file_name_2 = "Full_dataset/Lorentz/%scm/Lorentz_%scm_%sk_related_bit.csv" % (distance, distance, bit_rate) 
        file_name_3 = "Full_dataset/Lorentz/%scm/Lorentz_label_%scm_%sk.csv" % (distance, distance, bit_rate) 
        X1 = pd.read_csv(file_name_1, header = None)
        X1 = X1.values
        X2 = pd.read_csv(file_name_2, header = None)
        X2 = X2.values
        y = pd.read_csv(file_name_3, header = None)
        y = y.values
        y = y.reshape([len(y), ])
    return X1, X2, y