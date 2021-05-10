This program used for my thesis: Signal demodulation with neural network for OLED Visible Light Communication systems.
Author: Nguyen Tri Hieu
Github: hieunguyen810
Email: hieu.nguyennth123@hcmut.edu.vn
Report can be found at: "............."

### HOW TO RUN THIS PROGRAM #####
- Define at setup.json
    + Bitrate: from 10 to 400 kbps (missing at some bitrate)
    + Distance: from 10 to 40 cm (only use at 100kbps and 200kbps)
    + Preprocessing: Framing or DAE (denoising autoencoder)
    + enableCWT: true or false (cwt: Continuous wavelet transform)
- run file "run.py"