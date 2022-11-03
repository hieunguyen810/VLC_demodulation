This program used for my thesis: Signal demodulation with neural network for OLED Visible Light Communication systems.
<br />
Author: Nguyen Tri Hieu
<br />
Github: hieunguyen810
<br />
Email: hieu.nguyennth123@hcmut.edu.vn

----------------------------------------------------------------------
## GENERAL USAGE NOTES 
- Dataset for this program was measure on OVLC System.
- Input data: 2-level Lorentz pulse
- At each bitrate, we measure 40 times (250 bits per times)
- For PNN and GRNN, we have a default std at each scenerio, you can adjust it to get the higher performance. 
----------------------------------------------------------------------
## HOW TO RUN THIS PROGRAM
- Define at setup.json
    + Bitrate: from 10 to 400 kbps (missing at some bitrate)
    + Distance: from 10 to 40 cm (only use at 100kbps and 200kbps)
    + Preprocessing: Framing or DAE (denoising autoencoder)
    + enableCWT: true or false (cwt: Continuous wavelet transform)
    + Model: one of three model: Probabilistic neural networks, General regression neural networks, deep tensor neuarl networks (you can add new model at Model.py).
- run file "Main.py"
## Abstract
In this thesis, the two main parts are: measuring at laboratory 209B1 (Hcmut) and
building different machine learning models for signal demodulation. Data from the mea-
surement of two Lorentz pulses signals on visible light communication systems using
OLED use as datasets for the machine learning model. This thesis will detail the pro-
cesses from data acquisition, signal processing to signal classification.
<br />
The signal processing steps to be investigated include: reducing nonlinear distortion
with Denoising Autoencoder Neural Networks. The data to train is a two-level randomized
pulse with added Gaussian noise. The goal of the model is to learn the characteristics of a
two-level pulse to correct the distorted signal when passing through the VLC channel. The
second way to be investigated is to extract the signal into frames. Each frame contains
the classified bit along with the two preceding and two bits following it. In addition, a
time domain and frequency domain CWT technique is also used to improve BER.
<br />
To be able to classify signals with two different levels, we will use three methods:
PNN, GRNN and DTNN, then compare the three methods with each other and with the
demodulator using a threshold.
<br />
The goal of this thesis is to find the best method of preprocessing as well as classifying
signals that can reduce BER to a level below the allowable threshold in the VLC system
of 3.8e - 3, with the highest possible bit rate and distance.


