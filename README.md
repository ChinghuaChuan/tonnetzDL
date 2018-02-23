# tonnetzDL
Code for the paper “Modeling Temporal Tonal Relations in Polyphonic Music through Deep Networks with a Novel Image-Based Representation” (AAAI2018)

The code is created by Ching-Hua Chuan and Dorien Herremans as described in "Modeling Temporal Tonal Relations in Polyphonic Music
through Deep Networks with a Novel Image-Based Representation," in Proceedings of the 32nd AAAI Conference on Artificial Intelligence,
February 2-7, New Orleans, 2018. Part of the code is based on the Recurrent Neural Netowkrs in Tensorflow tutorial, 
https://r2rt.com/recurrent-neural-networks-in-tensorflow-ii.html This code is distributed under the GNU General Public License v3.0. 
If you have any questions regarding the code, please email to c.chuan@miami.edu

File tonnetz_dn.py
- First pre-train the CNN autoencoder and then train LSTM for 16-beat sequences
File tonnetz_dn_generate.py
- Generate music sequences given the first 16 beats using the trained model (output of tonnetz_dn.py)

Examples: (tested on Python2.7 and tensorflow 1.0)
$ python tonnetz_dn.py MuseData_train_tonnetz_vocab_p2.pickle MuseData_train_tonnetz_p2.pickle  --LSTMepoch 1

$ python tonnetz_dn_generate.py MuseData_train_tonnetz_vocab_p2.pickle MuseData_test_tonnetz_p2.pickle

$ python tonnetz_dn.py -h
usage: tonnetz_dn.py [-h] [--valid VALID] [--test TEST] [--CNNepoch CNNEPOCH]
                     [--LSTMepoch LSTMEPOCH]
                     vocab train

positional arguments:
  vocab                 vocabulary pickle file for tonnetz autoencoder
                        (Required)
  train                 pickle file for training (Required)

optional arguments:
  -h, --help            show this help message and exit
  --valid VALID         pickle file for validation
  --test TEST           pickle file for testing
  --CNNepoch CNNEPOCH   training epoch for CNN autoencoder
  --LSTMepoch LSTMEPOCH
                        training epoch for LSTM

The trained model (on MuseData and other pickle files in the examples) can be downloaded from: https://drive.google.com/open?id=1g_sxEKwrP9Wqt3YH99eEO1aXmmAqDbui


