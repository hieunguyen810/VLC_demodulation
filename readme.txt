This program used for my thesis: Signal demodulation with neural network for OLED Visible Light Communication systems.
Author: Nguyen Tri Hieu
Github: hieunguyen810
Email: hieu.nguyennth123@hcmut.edu.vn
Report can be found at: "............."

----------------------------------------------------------------------
### GENERAL USAGE NOTES ###
- Dataset for this program was measure on OVLC System.
- Input data: 2-level Lorentz pulse
- At each bitrate, we measure 40 times (250 bits per times)
- For PNN and GRNN, we have a default std at each scenerio, you can adjust it to get the higher performance. 
----------------------------------------------------------------------
### HOW TO RUN THIS PROGRAM #####
- Define at setup.json
    + Bitrate: from 10 to 400 kbps (supported Bitrate: 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 150, 200, 250, 300)
    + Distance: from 10 to 40 cm (only use at 100kbps and 200kbps)
    + Preprocessing: Framing or DAE (denoising autoencoder)
    + enableCWT: true or false (cwt: Continuous wavelet transform)
- run file "Main.py"
