# ==============================================================================
# The code is created by Ching-Hua Chuan and Dorien Herremans as described in
# "Modeling Temporal Tonal Relations in Polyphonic Music through Deep Networks with a Novel Image-Based Representation,"
# in Proceedings of the 32nd AAAI Conference on Artificial Intelligence, February 2-7, New Orleans, 2018.
# Part of the code is based on the Recurrent Neural Netowkrs in Tensorflow tutorial,
# https://r2rt.com/recurrent-neural-networks-in-tensorflow-ii.html
# This code is distributed under the GNU General Public License v3.0.
# If you have any questions regarding the code, please email to c.chuan@miami.edu
# ==============================================================================

from __future__ import division
import tensorflow as tf
import numpy as np
import pickle
from numpy import genfromtxt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('vocab', type=str, help='vocabulary pickle file for tonnetz autoencoder (Required)')
parser.add_argument('test', type=str, help='pickle file for testing (Required)')
args = parser.parse_args()
vocab_file = args.vocab
test_file = args.test

tonnetz_template = genfromtxt('tonnetz_template_row.txt', delimiter=',')

# autoencoder pre-training
def gen_data_cnn():
    l = pickle.load(file(vocab_file, 'rb'))
    return l

def gen_batch_cnn(raw_data, batch_size):
    tonnetz_seq = raw_data
    example = tonnetz_seq[0]
    total_num_examples = len(tonnetz_seq)
    num_batch = total_num_examples // batch_size

    for i in range(num_batch - 1):
        x = tonnetz_seq[batch_size * i:batch_size * (i + 1) - 1]
        yield x


def gen_epochs_cnn(n, batch_size):
    for i in range(n):
        yield gen_batch_cnn(gen_data_cnn(), batch_size+1)

# LSTM
def gen_data(infile):
    l = pickle.load(file(infile, 'rb'))
    return l

def gen_batch(raw_data, batch_size, num_steps):
    tonnetz_seq = raw_data
    example = tonnetz_seq[0]
    total_num_examples = len(tonnetz_seq)
    num_seqs = total_num_examples - (num_steps - 1)
    num_batch = num_seqs // batch_size
    for i in range(num_batch - 1):
        batch_x = []
        batch_y = []

        for j in range(batch_size):
            offset = (i * batch_size)
            sequence = tonnetz_seq[offset + j: offset + j + num_steps]
            batch_x = np.append(batch_x, sequence)
            sequence_y = tonnetz_seq[offset + j + num_steps + 1]
            batch_y = np.append(batch_y, sequence_y)
        x = np.reshape(batch_x, (-1, 288))
        y = np.reshape(batch_y, (-1, 288))
        yield x, y

def gen_epochs(n, batch_size, num_steps, file):
    for i in range(n):
        yield gen_batch(gen_data(file), batch_size, num_steps)

# setting the parameters for both networks
lstm_params = {}
lstm_params['num_steps'] = 16 # number of truncated backprop steps ('n' in the discussion above)
lstm_params['batch_size'] = 500
lstm_params['num_classes'] = 6*6*10
lstm_params['state_size'] = 6*6*10

autoencoder_params = {}
autoencoder_params['n_inputs'] = 12 * 24
autoencoder_params['n_hidden1'] = 20
autoencoder_params['n_hidden2'] = 10
autoencoder_params['n_hidden3'] = autoencoder_params['n_hidden1']
autoencoder_params['n_outputs'] = autoencoder_params['n_inputs']
autoencoder_params['learning_rate'] = 1e-4
autoencoder_params['n_epochs'] = 10

pred_params = {}
pred_params['policy'] = 'majority'
pred_params['threshold'] = 0.3
pred_params['pred_file'] = 'Testpred_' + pred_params['policy'] + str(pred_params['threshold'])+'.pickle'
pred_params['groundtruth_file'] = 'Testgroundtruth_' + pred_params['policy'] + str(pred_params['threshold']) + '.pickle'

def generate_characters(checkpoint):

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())
        meta_file = checkpoint + '.meta'
        saver = tf.train.import_meta_graph(meta_file)
        saver.restore(sess, checkpoint)

        print("Restoring variables...")

        graph = tf.get_default_graph()

        # restore the learned weights/biases needed for autoencoder
        x = graph.get_tensor_by_name("input_placeholder:0")  # placeholder
        y = graph.get_tensor_by_name("output_placeholder:0")  # placeholder

        weight1 = sess.run(graph.get_tensor_by_name("weight1:0"))  # a variable
        bias1 = sess.run(graph.get_tensor_by_name("bias1:0"))
        weight2 = sess.run(graph.get_tensor_by_name("weight2:0"))
        bias2 = sess.run(graph.get_tensor_by_name("bias2:0"))
        final_w = sess.run(graph.get_tensor_by_name("final_w:0"))
        final_b = sess.run(graph.get_tensor_by_name("final_b:0"))
        fixed_w1 = graph.get_tensor_by_name("fixed_w1:0")
        fixed_b1 = graph.get_tensor_by_name("fixed_b1:0")
        fixed_w2 = graph.get_tensor_by_name("fixed_w2:0")
        fixed_b2 = graph.get_tensor_by_name("fixed_b2:0")
        fixed_final_w = graph.get_tensor_by_name("fixed_final_w:0")
        fixed_final_b = graph.get_tensor_by_name("fixed_final_b:0")
        predictions = graph.get_tensor_by_name("predictions:0")
        valid_loss = graph.get_tensor_by_name("valid_loss:0")

        out_pred = []
        out_groundtruth = []

        for id, epoch in enumerate(gen_epochs(1, lstm_params['batch_size'], lstm_params['num_steps'], test_file)):
            step = 0
            test_total_loss = 0
            all_predictions = np.empty(shape=[0, 288])

            for xi, yi in epoch:
                step +=1
                feed_dict = {x: xi, y: yi, fixed_w1: weight1, fixed_b1: bias1, fixed_w2: weight2,
                             fixed_b2: bias2, fixed_final_w: final_w, fixed_final_b: final_b}
                pred, loss = sess.run([predictions, valid_loss], feed_dict)
                test_total_loss += loss
                print("batch ", step, " loss = ", loss)
                all_predictions = np.append(all_predictions, pred, axis=0)
                # getting pitch output for the batch
                out_pred.append(tonnetz2note(pred, pred_params['policy'], pred_params['threshold']))
                out_groundtruth.append(tonnetz2note(yi, pred_params['policy'], pred_params['threshold']))

            print('The average test loss is: ', test_total_loss/step)

        #np.savetxt('MuseData_tonnetz_test_predictions.txt', all_predictions)

        print("Writing generated sequences in file:", pred_params['pred_file'])
        with open(pred_params['pred_file'], 'wb') as f:
            pickle.dump(out_pred, f)

        print("Writing the groundtruth for the generated sequences in file:", pred_params['groundtruth_file'])
        with open(pred_params['groundtruth_file'], 'wb') as f:
            pickle.dump(out_groundtruth, f)

#policy = 'max'
#threshold = 0.3
pitch_range = [6, 109] # tonnetz pitch range
def tonnetz2note(pred, policy, threshold):
    pitch_out_batch = []
    count = 0
    for frame in pred:
        pitch = pitch_range[0]
        pitch_out = []
        while pitch <= pitch_range[1]:
            index = np.where(tonnetz_template == pitch)
            pitch_preds = []
            for loc in index:
                for i in loc:
                    pitch_preds = np.append(pitch_preds, frame[i])

            pitch_preds = np.array(pitch_preds)
            out = 0
            if policy == 'mean':
                if np.mean(pitch_preds) > threshold:
                    out = 1
            elif policy == 'majority':
                index_count = np.where(pitch_preds > threshold)
                if len(index_count) > (len(pitch_preds)/2):
                    out = 1
            elif policy == 'min':
                if min(pitch_preds) > threshold:
                    out = 1
            else:   # max as default
                if max(pitch_preds) > threshold:
                    out = 1

            if out == 1:
                pitch_out = np.append(pitch_out, pitch)
            pitch +=1

        pitch_out_batch.append(pitch_out)
        count +=1
    return pitch_out_batch

print ("Generating sequences for test file:", test_file)
generate_characters("./save/tonnetz_training_model")



