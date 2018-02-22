# tonnetzDL
Code for the paper “Modeling Temporal Tonal Relations in Polyphonic Music through Deep Networks with a Novel Image-Based Representation” (AAAI2018)
# ==============================================================================
# The code is created by Ching-Hua Chuan and Dorien Herremans as described in
# "Modeling Temporal Tonal Relations in Polyphonic Music through Deep Networks with a Novel Image-Based Representation,"
# in Proceedings of the 32nd AAAI Conference on Artificial Intelligence, February 2-7, New Orleans, 2018.
# Part of the code is based on the Recurrent Neural Netowkrs in Tensorflow tutorial,
# https://r2rt.com/recurrent-neural-networks-in-tensorflow-ii.html
# This code is distributed under the GNU General Public License v3.0.
# If you have any questions regarding the code, please email to c.chuan@miami.edu
# ==============================================================================

File tonnetz_dn.py
- First pre-train the CNN autoencoder and then train LSTM for 16-beat sequences

File tonnetz_dn_generate.py
- Generate music sequences given the first 16 beats using the trained model (output of tonnetz_dn.py)

The trained model (on MuseData) can be downloaded from: https://drive.google.com/open?id=1g_sxEKwrP9Wqt3YH99eEO1aXmmAqDbui


