import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
def show_result(test_loss, val_loss, acc, val_acc, num_epochs):
    epochs_range = range(num_epochs)
    _, (ax1, ax2) = plt.subplots(2)
    ax1.plot(epochs_range, test_loss, label='Test Loss')
    ax1.plot(epochs_range, val_loss, label='Validation loss')
    ax1.legend(loc='upper right')
    ax1.set_title('Loss')
    ax2.plot(epochs_range, acc, label='Test accuracy')
    ax2.plot(epochs_range, val_acc, label='Validation accuracy')
    ax2.set_title('Accuracy')
    ax2.legend(loc='upper right')
    plt.show()
def show_signal():
    X = pd.read_csv("Lorentz_dataset/Lorentz_20cm_250k.csv", header = None)
    y = pd.read_csv("Lorentz_dataset/Lorentz_label_20cm_250k.csv", header = None)
    X = X.values
    X = X.reshape(-1)
    y = y.values
    y = y.reshape(-1)
    y = np.kron(y, [1, 1, 1, 1, 1])
    y = - y + 1
    plt.style.use('dark_background')
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    #ax.plot(y[10:500]*500, label = "Transmitted signal", color = "gold")
    ax.plot(X[0:200], label = "Received signal", color = "gold")
    minor_ticks = np.arange(0, 200, 5)
    ax.set_xticks(minor_ticks, minor=True)
    #ax.set_yticks(minor_ticks, minor=True)
    ax.grid(which='both')
    ax.grid(which='minor', alpha=0.2)
    plt.ylabel("Time")
    plt.xlabel("Amplitude")
    #ax.legend()
    plt.show()
def show_hist():
    X = pd.read_csv("Lorentz_dataset/Lorentz_20cm_250k_related_bit.csv", header = None)
    y = pd.read_csv("Lorentz_dataset/Lorentz_label_20cm_250k.csv", header = None)
    X = X.values
    y = y.values
    label_1 = 0
    label_0 = 0
    class_0 = []
    class_1 = []
    for j in np.arange(len(y)):
        if y[j] == 1:
            label_1+=1
            class_1 = np.append(class_1, X[j])
        else:
            label_0+=1
            class_0 = np.append(class_0, X[j])
    plt.style.use('dark_background')
    plt.hist(class_1, alpha = 0.7, label = "Low level", color = "gold")
    plt.hist(class_0, alpha = 0.7, label = "High level", color = "lawngreen")
    plt.xlabel("Ampitude")
    plt.ylabel("Number of occurrences")
    plt.legend()
    plt.show()
def chart():
    X = pd.read_csv("Lorentz_dataset/20cm/BER_chart_20cm.csv")
    X = X.values
    X = X[:, 4]
    y = np.arange(10, 510, 10)
    plt.plot(y, X, label = "BER Estimate")
    plt.ylabel("Bit rate")
    plt.xlabel("BER")
    plt.legend()
    plt.show()  
def main():
    show_hist()
if __name__ == '__main__':
    main() 