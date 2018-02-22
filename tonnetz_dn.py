# ==============================================================================
# The code is created by Ching-Hua Chuan and Dorien Herremans as described in
# "Modeling Temporal Tonal Relations in Polyphonic Music through Deep Networks with a Novel Image-Based Representation,"
# in Proceedings of the 32nd AAAI Conference on Artificial Intelligence, February 2-7, New Orleans, 2018.
# Part of the code is based on the Recurrent Neural Netowkrs in Tensorflow tutorial,
# https://r2rt.com/recurrent-neural-networks-in-tensorflow-ii.html
# This code is distributed under the GNU General Public License v3.0.
# If you have any questions regarding the code, please email to c.chuan@miami.edu
# ==============================================================================

import tensorflow as tf
import numpy as np
import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('vocab', type=str, help='vocabulary pickle file for tonnetz autoencoder (Required)')
parser.add_argument('train', type=str, help='pickle file for training (Required)')
parser.add_argument('--valid', type=str, help='pickle file for validation')
parser.add_argument('--test', type=str, help='pickle file for testing')
parser.add_argument('--CNNepoch', type=int, help='training epoch for CNN autoencoder')
parser.add_argument('--LSTMepoch', type=int, help='training epoch for LSTM')
args = parser.parse_args()

vocab_file = args.vocab
train_file = args.train

if args.valid is not None:
    valid_flag = True
    valid_file = args.valid
else:
    valid_flag = False

if args.test is not None:
    test_flag = True
    test_file = args.test
else:
    test_flag = False

if args.CNNepoch is not None:
    cnn_epoch = args.CNNepoch
else:
    cnn_epoch = 10

if args.LSTMepoch is not None:
    lstm_epoch = args.LSTMepoch
else:
    lstm_epoch = 27

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
        yield gen_batch_cnn(gen_data_cnn(), batch_size + 1)


# LSTM
def gen_data(infile):
    if 'txt' in infile:
        f = open(infile, 'r')
        l = [list(map(float, line.split(','))) for line in f]
    else:
        l = pickle.load(file(infile, 'rb'))
    return l


def gen_batch(raw_data, batch_size, num_steps):
    tonnetz_seq = raw_data
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


def gen_epochs(n, batch_size, num_steps, infile):
    for i in range(n):
        yield gen_batch(gen_data(infile), batch_size, num_steps)


def reset_graph():
    if 'sess' in globals() and sess:
        sess.close()
    tf.reset_default_graph()


def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')


def max_pool(x, k1, k2):
    return tf.nn.max_pool(x, ksize=[1, k1, k2, 1], strides=[1, k1, k2, 1], padding='SAME')


# def autoencoder():
reset_graph()

# setting the parameters for both networks
lstm_params = {}
lstm_params['num_steps'] = 16  # number of truncated backprop steps ('n' in the discussion above)
lstm_params['batch_size'] = 500
lstm_params['num_classes'] = 6 * 6 * 10
lstm_params['state_size'] = 6 * 6 * 10

autoencoder_params = {}
autoencoder_params['n_inputs'] = 12 * 24
autoencoder_params['n_hidden1'] = 20
autoencoder_params['n_hidden2'] = 10
autoencoder_params['n_hidden3'] = autoencoder_params['n_hidden1']
autoencoder_params['n_outputs'] = autoencoder_params['n_inputs']
autoencoder_params['learning_rate'] = 1e-4
autoencoder_params['n_epochs'] = cnn_epoch


# building the computational graph
def build_graph(cell_type=None,
                num_weights_for_custom_cell=5,
                state_size=lstm_params['state_size'],
                num_classes=lstm_params['num_classes'],
                batch_size=lstm_params['batch_size'],
                num_steps=lstm_params['num_steps'],
                num_layers=3,
                build_with_dropout=False,
                learning_rate=1e-4,
                autoencoder_params=autoencoder_params):
    initializer = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG", uniform=True)

    x = tf.placeholder(tf.float32, [None, autoencoder_params['n_inputs']],
                       name='input_placeholder')
    y = tf.placeholder(tf.float32, [None, autoencoder_params['n_inputs']], name="output_placeholder")
    dropout = tf.constant(1.0)
    x_image = tf.reshape(x, shape=[-1, 24, 12, 1])

    weight1 = tf.Variable(initializer([3, 3, 1, autoencoder_params['n_hidden1']]), name='weight1')
    bias1 = tf.Variable(tf.zeros(autoencoder_params['n_hidden1']), name='bias1')
    hidden1 = tf.nn.relu(conv2d(x_image, weight1) + bias1)
    pool1 = max_pool(hidden1, 2, 2)

    weight2 = tf.Variable(initializer([3, 3, autoencoder_params['n_hidden1'], autoencoder_params['n_hidden2']]),
                          name='weight2')
    bias2 = tf.Variable(tf.zeros(autoencoder_params['n_hidden2']), name='bias2')
    hidden2 = tf.nn.relu(conv2d(pool1, weight2) + bias2)
    pool2 = max_pool(hidden2, 2, 1)

    # a fully connected layer for reduced-dimension CNN
    pool2_reshaped = tf.reshape(pool2, shape=[-1, 6 * 6 * autoencoder_params['n_hidden2']])
    final_w = tf.Variable(initializer([6 * 6 * autoencoder_params['n_hidden2'], autoencoder_params['n_outputs']]),
                          name='final_w')
    final_b = tf.Variable(tf.zeros(autoencoder_params['n_outputs']), name='final_b')
    final_o = tf.matmul(pool2_reshaped, final_w) + final_b

    autoencoder_outputs = final_o

    loss_cnn = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=x, logits=autoencoder_outputs))

    optimizer_cnn = tf.train.AdamOptimizer(autoencoder_params['learning_rate'])
    training_op_cnn = optimizer_cnn.minimize(loss_cnn)

    # LSTM
    fixed_w1 = tf.placeholder(tf.float32, [3, 3, 1, autoencoder_params['n_hidden1']], name='fixed_w1')
    fixed_b1 = tf.placeholder(tf.float32, [autoencoder_params['n_hidden1']], name='fixed_b1')
    fixed_w2 = tf.placeholder(tf.float32, [3, 3, autoencoder_params['n_hidden1'], autoencoder_params['n_hidden2']],
                              name='fixed_w2')
    fixed_b2 = tf.placeholder(tf.float32, [autoencoder_params['n_hidden2']], name='fixed_b2')
    fixed_final_w = tf.placeholder(tf.float32,
                                   [6 * 6 * autoencoder_params['n_hidden2'], autoencoder_params['n_outputs']],
                                   name='fixed_final_w')
    fixed_final_b = tf.placeholder(tf.float32, [autoencoder_params['n_outputs']], name='fixed_final_b')

    lstm_hidden1 = tf.nn.relu(conv2d(x_image, fixed_w1) + fixed_b1)
    lstm_pool1 = max_pool(lstm_hidden1, 2, 2)
    lstm_hidden2 = tf.nn.relu(conv2d(lstm_pool1, fixed_w2) + fixed_b2)
    lstm_pool2 = max_pool(lstm_hidden2, 2, 1)
    lstm_pool2_reshaped = tf.reshape(lstm_pool2, shape=[-1, 6 * 6 * autoencoder_params['n_hidden2']])

    rnn_inputs = tf.reshape(lstm_pool2_reshaped,
                            shape=[-1, lstm_params['num_steps'], 6 * 6 * autoencoder_params['n_hidden2']])

    if cell_type == 'GRU':
        cell = tf.contrib.rnn.GRUCell(state_size)
    elif cell_type == 'LSTM':
        cell = tf.contrib.rnn.LSTMCell(state_size, state_is_tuple=True)
    elif cell_type == 'LN_LSTM':
        cell = LayerNormalizedLSTMCell(state_size)
    else:
        cell = tf.contrib.rnn.BasicRNNCell(state_size)

    if build_with_dropout:
        cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=dropout)

    if cell_type == 'LSTM' or cell_type == 'LN_LSTM':
        cell = tf.contrib.rnn.MultiRNNCell([build_rnn_cell('LSTM', state_size) for _ in range(num_layers)],
                                           state_is_tuple=True)
    else:
        cell = tf.contrib.rnn.MultiRNNCell([build_rnn_cell('Basic', state_size) for _ in range(num_layers)],
                                           state_is_tuple=True)

    if build_with_dropout:
        cell = tf.contrib.DropoutWrapper(cell, output_keep_prob=dropout)

    init_state = cell.zero_state(batch_size, tf.float32)
    rnn_outputs, final_state = tf.nn.dynamic_rnn(cell, rnn_inputs, initial_state=init_state)


    with tf.variable_scope('softmax'):
        W = tf.get_variable('W', [state_size, num_classes])
        b = tf.get_variable('b', [num_classes], initializer=tf.constant_initializer(0.0))

    rnn_outputs = rnn_outputs[:, num_steps - 1, :]

    y_image = tf.reshape(y, shape=[-1, 24, 12, 1])
    y_hidden1 = tf.nn.relu(conv2d(y_image, fixed_w1) + fixed_b1)
    y_pool1 = max_pool(y_hidden1, 2, 2)
    y_hidden2 = tf.nn.relu(conv2d(y_pool1, fixed_w2) + fixed_b2)
    y_pool2 = max_pool(y_hidden2, 2, 1)

    y_pool2_reshaped = tf.reshape(y_pool2, shape=[-1, 6 * 6 * autoencoder_params['n_hidden2']])
    rnn_outputs_reshaped = tf.reshape(rnn_outputs, [-1, num_classes])
    logits = tf.matmul(rnn_outputs_reshaped, W) + b

    total_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=y_pool2_reshaped))

    # .............
    # decode the output and compare it with original y

    decode_logits = tf.matmul(rnn_outputs_reshaped, fixed_final_w) + fixed_final_b

    tonnetz_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=decode_logits, labels=y))
    valid_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=decode_logits, labels=y),
                                name='valid_loss')
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(tonnetz_loss)
    predictions = tf.nn.sigmoid(decode_logits, name='predictions')

    return dict(
        x=x,
        y=y,
        init_state=init_state,
        final_state=final_state,
        total_loss=total_loss,
        train_step=train_step,
        tonnetz_loss=tonnetz_loss,
        valid_loss=valid_loss,
        preds=predictions,
        training_op_cnn=training_op_cnn,
        loss_cnn=loss_cnn,
        autoencoder_outputs=autoencoder_outputs,
        final_w=final_w,
        weight2=weight2,
        weight1=weight1,
        bias1=bias1,
        bias2=bias2,
        final_b=final_b,
        fixed_w1=fixed_w1,
        fixed_b1=fixed_b1,
        fixed_w2=fixed_w2,
        fixed_b2=fixed_b2,
        fixed_final_w=fixed_final_w,
        fixed_final_b=fixed_final_b,
        saver=tf.train.Saver()
    )


def build_rnn_cell(cell_type, state_size):
    if cell_type == 'GRU':
        cell = tf.contrib.rnn.GRUCell(state_size)
    elif cell_type == 'LSTM':
        cell = tf.contrib.rnn.LSTMCell(state_size, state_is_tuple=True)
    elif cell_type == 'LN_LSTM':
        cell = LayerNormalizedLSTMCell(state_size)
    else:
        cell = tf.contrib.rnn.BasicRNNCell(state_size)
    return cell


def train_network(g, num_epochs, num_steps, batch_size=32, verbose=True, save=True):
    tf.set_random_seed(2345)
    with tf.Session() as sess:
        with tf.device('/cpu:0'):
            sess.run(tf.global_variables_initializer())

            # pre_training = True
            if (True):
                # pre-training
                print("CNN autoencoder pre-training starts...")
                for idx, batch in enumerate(gen_epochs_cnn(autoencoder_params['n_epochs'], lstm_params['batch_size'])):
                    training_losses = 0
                    print("epoch", idx)
                    step = 0
                    for x_batch in batch:
                        step = step + 1
                        feed_dict = {g['x']: x_batch}
                        _, b_loss, batch_outputs, final_w, weight2, weight1, final_b, bias2, bias1 = sess.run(
                            [g['training_op_cnn'],
                             g['loss_cnn'],
                             g['autoencoder_outputs'],
                             g['final_w'],
                             g['weight2'],
                             g['weight1'],
                             g['final_b'],
                             g['bias2'],
                             g['bias1']],
                            feed_dict)
                        #print("batch", step, ", batch loss:", b_loss)
                        training_losses = training_losses + b_loss
                        if idx == autoencoder_params['n_epochs'] - 1:
                            if step == 1:
                                final_outputs = batch_outputs
                            else:
                                final_outputs = np.append(final_outputs, batch_outputs, axis=0)

                    training_losses = training_losses / step
                    print("epoch average loss:", training_losses)


            # LSTM: using the input from CNN
            training_losses = []
            batch_losses = []
            valid_losses = []
            print("LSTM training starts...")
            for idx, epoch in enumerate(gen_epochs(num_epochs, batch_size, num_steps, train_file)):
                print("epoch", idx)
                sum_tonnetz_loss = 0
                steps = 0
                training_state = None
                for X, Y in epoch:
                    steps += 1

                    feed_dict = {g['x']: X, g['y']: Y, g['fixed_w1']: weight1, g['fixed_b1']: bias1,
                                 g['fixed_w2']: weight2,
                                 g['fixed_b2']: bias2, g['fixed_final_w']: final_w, g['fixed_final_b']: final_b}
                    if training_state is not None:
                        feed_dict[g['init_state']] = training_state
                    training_state, _, tonnetz_loss, final_w, weight2, fixed_final_w = sess.run([g['final_state'],
                                                                                                 g['train_step'],
                                                                                                 g['tonnetz_loss'],
                                                                                                 g['final_w'],
                                                                                                 g['weight2'],
                                                                                                 g['fixed_final_w']],
                                                                                                feed_dict)
                    # training_loss += training_loss_
                    sum_tonnetz_loss += tonnetz_loss
                    print(steps - 1, tonnetz_loss)
                    batch_losses.append(tonnetz_loss)

                if verbose:
                    print("Average training loss for Epoch", idx, ":", sum_tonnetz_loss / steps)
                training_losses.append(sum_tonnetz_loss / steps)

                ##validation after each epoch:
                if valid_flag == True:
                    for idv, epochv in enumerate(gen_epochs(1, batch_size, num_steps, valid_file)):
                        step = 0
                        valid_total_loss =0
                        for xv, yv in epochv:
                            step += 1
                   #        print("validation step:%d" % step)
                            feed_dict = {g['x']: xv, g['y']:yv, g['fixed_w1']:weight1, g['fixed_b1']:bias1, g['fixed_w2']:weight2, g['fixed_b2']:bias2, g['fixed_final_w']:final_w, g['fixed_final_b']:final_b}
                            loss = sess.run(g['valid_loss'], feed_dict)
                   #        print("validation loss: %f" % loss)
                            valid_total_loss += loss
                        valid_losses.append(valid_total_loss/step)
                    print("Average validation loss for Epoch", idx, ":", valid_total_loss / step)

            if test_flag == True:
                for idx, epoch in enumerate(gen_epochs(1, batch_size, num_steps, test_file)):
                    step = 0
                    test_total_loss = 0
                    for x, y in epoch:
                        step += 1

                        feed_dict = {g['x']: x, g['y']: y, g['fixed_w1']: weight1, g['fixed_b1']: bias1,
                                     g['fixed_w2']: weight2, g['fixed_b2']: bias2, g['fixed_final_w']: final_w,
                                     g['fixed_final_b']: final_b}
                        loss = sess.run(g['valid_loss'], feed_dict)
                        print("test step:", step, "with loss", loss)
                        test_total_loss += loss
                    print("Average test loss =", test_total_loss/step)

        if isinstance(save, str):
            g['saver'].save(sess, save)





    return training_losses

g = build_graph(cell_type='LSTM', num_steps=lstm_params['num_steps'])
losses = train_network(g, lstm_epoch, num_steps=lstm_params['num_steps'], batch_size=lstm_params['batch_size'], save="save/tonnetz_training_model")

