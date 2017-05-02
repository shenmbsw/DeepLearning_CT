import tensorflow as tf
import numpy as np
import data_generators as dg
import small_data_generators as sdg

def cnn_rnn(X):
    pad_heights = 470
    n_hidden_dim = 200
    batch_size = X.shape[0]
    LSTM = tf.contrib.rnn.LSTMCell(num_units = n_hidden_dim)
    initial_state = state = LSTM.zero_state(batch_size,tf.float32)
    X_ = tf.expand_dims(X, 4)
    x = tf.layers.average_pooling3d(inputs=X_, pool_size=[1,2,2],strides=[1,2,2], padding="valid")
    #print(x.shape)
    for t in range(pad_heights):
        x_ = x[:,t,:,:,:]
        with tf.variable_scope("cnn",reuse=False):
            if t>0:
                tf.get_variable_scope().reuse_variables()
            conv1 = tf.layers.conv2d(inputs=x_, filters=16, strides=(3, 3), kernel_size=[5, 5], padding="same", activation=tf.nn.relu, name='conv1')
            pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[3, 3], strides=2, name='pool1')
            conv2 = tf.layers.conv2d(inputs=pool1, filters=16, strides=(3, 3), kernel_size=[5, 5], padding="same", activation=tf.nn.relu,name='conv2')
            pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[3, 3], strides=2, name='pool2')
            pool_flat = tf.contrib.layers.flatten(pool2, scope='poolflat')
        with tf.variable_scope("rnn", reuse=False):
            if t>0:
                tf.get_variable_scope().reuse_variables()
            RNN_output, state = LSTM(pool_flat, state)
    RNN_dropout = tf.layers.dropout(inputs=RNN_output, rate=0.5)
    dense = tf.layers.dense(inputs=RNN_dropout, units=50, activation=tf.nn.relu)
    logits = tf.layers.dense(inputs=dense, units=1)
    return logits

def apply_classification_loss_rnn(model_function):
    with tf.Graph().as_default() as g:
        with tf.device("/gpu:0"):
            trainer = tf.train.AdamOptimizer()
            x_ = tf.placeholder(tf.float32, [1, 470, 512, 512])
            y_ = tf.placeholder(tf.int32, [1, 1])
            y_logits = model_function(x_)
            loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(y_, tf.float32),logits=y_logits)
            cross_entropy_loss = tf.reduce_mean(loss)
            train_op = trainer.minimize(cross_entropy_loss)
            y_prob = tf.sigmoid(y_logits)
            y_pred = tf.cast(tf.round(y_prob), tf.int32)
            correct_prediction = tf.equal(y_pred, y_)
            accuracy = tf.cast(correct_prediction, tf.float32)
    model_dict = {'graph': g, 'inputs': [x_, y_], 'train_op': train_op,
                  'pred':y_pred,'prob':y_prob,
                  'accuracy': accuracy, 'loss': cross_entropy_loss}
    return model_dict

def train_model(model_dict, dataset_generators, epoch_n, print_every):
    with model_dict['graph'].as_default(), tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        test_record = ()
        for epoch_i in range(epoch_n):
            collect_loss= []
            collect_acc = []
            for iter_i, data_batch in enumerate(dataset_generators['train']):
                train_feed_dict = dict(zip(model_dict['inputs'], data_batch))
                train_to_compute = [model_dict['train_op'],model_dict['loss'],model_dict['accuracy']]
                _, loss, acc = sess.run(train_to_compute, feed_dict=train_feed_dict)
                collect_loss.append(loss)
                collect_acc.append(acc)
            print(np.mean(collect_loss, axis=0),np.mean(collect_acc, axis=0))

            collect_arr = []
            for a, test_batch in enumerate(dataset_generators['test']):
                test_feed_dict = dict(zip(model_dict['inputs'], test_batch))
                to_compute = [model_dict['loss'], model_dict['accuracy']]
                collect_arr.append(sess.run(to_compute, test_feed_dict))
            test_averages = np.mean(collect_arr, axis=0)
            test_record += tuple(test_averages)

            fmt = (epoch_i,) + tuple(test_averages)

            print('epoch {:d},  loss: {:.3f}, '
                  'accuracy: {:.3f}'.format(*fmt))

        print('the_test_record_is')
        for i in range(len(test_record)//2):
            print(test_record[2*i], test_record[2*i+1])
        for a, test_batch in enumerate(dataset_generators['test']):
            test_feed_dict = dict(zip(model_dict['inputs'], test_batch))
            print(sess.run(model_dict['pred'], test_feed_dict))

def main():
    train_data_filepath = 'processed_train_data/train_data'
    train_labels_filepath = 'processed_train_data/train_labels'
    test_data_filepath = 'processed_test_data/test_data'
    test_labels_filepath = 'processed_test_data/test_labels'

    train_data_pathlist, train_labels_pathlist = dg.creat_datapath(train_data_filepath, train_labels_filepath, 1)
    test_data_pathlist, test_labels_pathlist = dg.creat_datapath(test_data_filepath, test_labels_filepath, 1)

    dataset_generators = {
            'train': dg.dataset_2d_iterator(train_data_pathlist, train_labels_pathlist),
            'test':  dg.dataset_2d_iterator(test_data_pathlist, test_labels_pathlist),
        }

    #
    # data_filepath = 'temp.npy'
    # label_filepath = 'label.npy'
    # dataset_generators = {
    #         'train': sdg.dataset_2d_iterator(data_filepath, label_filepath),
    #         'test':  sdg.dataset_2d_iterator(data_filepath, label_filepath),
    #     }
    model_dict = apply_classification_loss_rnn(cnn_rnn)
    train_model(model_dict, dataset_generators, epoch_n = 2, print_every = 1)

main()
